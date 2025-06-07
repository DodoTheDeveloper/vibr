const std = @import("std");
const onnx = @cImport({
    @cInclude("onnxruntime_c_api.h");
});
const ModelInfo = @import("./ModelInfo.zig").ModelInfo;
const onnx_core = @import("./core.zig");

pub const OnnxError = error{ UnwrapOnnxFnFoundNull, UnexpectedNull, UnsupportedTensorDataType };

// Certain tensor dimensions are dynamic (e.g. batch size) which is
// indicated by a -1 value. We replace those values with a fixed 1.
pub fn replaceDynamicTensorDims(dims: *const []i64) void {
    for (dims.*, 0..) |dim, idx| {
        if (dim == -1) dims.*[idx] = 1;
    }
}

// Returns the product of the sizes of each dimension of a tensor. Such a slice looks
// typically looks like [batch, heads, seq, head_dim]
// As the batch dimmension & past sequence length can be dynamic (-1) we must
// replace them with actual fixed values which are currenty hardcoded to 1.
pub fn getTensorDimProduct(dims: []const i64) usize {
    var total_tims: i64 = 1;

    for (dims) |dim| {
        if (dim == -1) total_tims *= 1 else total_tims *= dim;
    }
    return @intCast(total_tims);
}

// TODO release memory when fetching names from inputs from the channels!! OrtAllocator
pub fn parseModelInfo(allocator: std.mem.Allocator, onnx_allocator: *onnx.OrtAllocator, api: *const onnx.OrtApi, session: *const onnx.OrtSession) !*ModelInfo {
    // inputs
    var num_inputs: usize = 0;
    try onnx_core.SessionGetInputCount(api, session, &num_inputs);

    const inputs: []ModelInfo.Input = try allocator.alloc(ModelInfo.Input, num_inputs);
    for (0..num_inputs) |input_idx| {
        var input_name_arr: [*:0]u8 = undefined;
        try onnx_core.SessionGetInputName(api, session, input_idx, onnx_allocator, &input_name_arr);



        var type_info: ?*onnx.OrtTypeInfo = null;
        try onnx_core.SessionGetInputTypeInfo(api, session, input_idx, &type_info);
        if (type_info == null) return OnnxError.UnexpectedNull;

        const input_name_span: []const u8 = std.mem.span(input_name_arr);
        const input_name_dup = try allocator.dupeZ(u8, input_name_span);

        var tensor_info: ?*const onnx.OrtTensorTypeAndShapeInfo = null;
        try onnx_core.CastTypeInfoToTensorInfo(api, type_info, &tensor_info);
        if (tensor_info == null) return OnnxError.UnexpectedNull;

        var element_type: onnx.ONNXTensorElementDataType = undefined;
        try onnx_core.GetTensorElementType(api, tensor_info, &element_type);

        var dim_count: usize = 0;
        try onnx_core.GetDimensionsCount(api, tensor_info, &dim_count);

        const dim_values: []i64 = try allocator.alloc(i64, dim_count);
        //defer allocator.free(dim_values);

        try onnx_core.GetDimensions(api, tensor_info, dim_values.ptr, dim_count);
        replaceDynamicTensorDims(&dim_values);

        const is_cached: bool = std.mem.startsWith(u8, input_name_dup, "past_key_values");

        inputs[input_idx] = .{ .name = input_name_dup, .data_type = element_type, .dimensions = dim_values, .is_cached = is_cached };
        std.debug.print("[dodo] idx:{d} name {s} arr {*}\n", .{input_idx, input_name_span, input_name_arr});
    }

    // outputs
    var num_outputs: usize = 0;
    try onnx_core.SessionGetOutputCount(api,session, &num_outputs);

    const outputs: []ModelInfo.Output = try allocator.alloc(ModelInfo.Output, num_outputs);

    for (0..num_outputs) |output_idx| {
        var output_name_arr: [*:0]u8 = undefined;
        try onnx_core.SessionGetOutputName(api, session, output_idx, onnx_allocator, &output_name_arr);

        var type_info: ?*onnx.OrtTypeInfo = null;
        try onnx_core.SessionGetOutputTypeInfo(api, session, output_idx, &type_info);
        if (type_info == null) return OnnxError.UnexpectedNull;

        const output_name_span: []const u8 = std.mem.span(output_name_arr);
        const output_name_dup = try allocator.dupeZ(u8, output_name_span);

        var tensor_info: ?*const onnx.OrtTensorTypeAndShapeInfo = null;
        try onnx_core.CastTypeInfoToTensorInfo(api, type_info, &tensor_info);
        if (tensor_info == null) return OnnxError.UnexpectedNull;

        var element_type: onnx.ONNXTensorElementDataType = undefined;
        try onnx_core.GetTensorElementType(api, tensor_info, &element_type);

        var dim_count: usize = 0;
        try onnx_core.GetDimensionsCount(api, tensor_info, &dim_count);

        const dim_values: []i64 = try allocator.alloc(i64, dim_count);
        //defer allocator.free(dim_values);

        try onnx_core.GetDimensions(api, tensor_info, dim_values.ptr, dim_count);
        replaceDynamicTensorDims(&dim_values);

        const is_cached: bool = std.mem.startsWith(u8, output_name_dup, "past_key_values");

        outputs[output_idx] = .{ .name = output_name_dup, .data_type = element_type, .dimensions = dim_values, .is_cached = is_cached };
    }

    const model_info = try ModelInfo.create(allocator, inputs, outputs);
    return model_info;
}

test "replaceDynamicTensorDims replaces -1 dims with 1" {
    const given_allocator = std.testing.allocator;
    //given_dims.ptr = &.{ -1, 2, 3, -1, 5 };
    const given_dims = try given_allocator.dupe(i64, &.{ -1, 2, 3, -1, 5 });
    defer given_allocator.free(given_dims);

    const expected_dims: []const i64 = &.{ 1, 2, 3, 1, 5 };
    replaceDynamicTensorDims(&given_dims);
    const actual_dims = given_dims;

    try std.testing.expectEqualSlices(i64, expected_dims, actual_dims);
}

test "getTensorTotalDimSize calculates dims product" {
    const given_dims: []const i64 = &.{ 1, 2, 3, 4, 5 };
    const expected_dims = 2 * 3 * 4 * 5;
    const actual_total = getTensorDimProduct(given_dims);
    try std.testing.expectEqual(expected_dims, actual_total);
}

test "getTensorTotalDimSize sets dynamic (-1) to 1" {
    const given_dims: []const i64 = &.{ -1, 2, 3, -1, 5 };
    const expected_dims = 2 * 3 * 5;
    const actual_total = getTensorDimProduct(given_dims);
    try std.testing.expectEqual(expected_dims, actual_total);
}



test "parseModelInfo with with_past_kv.onnx model" {
    const allocator = std.testing.allocator;
    
    // Initialize ONNX runtime
    const base: *const onnx.OrtApiBase = try onnx_core.OrtGetApiBase();
    const api: *const onnx.OrtApi = try onnx_core.GetApi(base);
    
    var onnx_allocator: *onnx.OrtAllocator = undefined;
    try onnx_core.GetAllocatorWithDefaultOptions(api, &onnx_allocator);
    
    // Create environment
    var env: *onnx.OrtEnv = undefined;
    try onnx_core.CreateEnv( api, onnx.ORT_LOGGING_LEVEL_WARNING, "test_env", &env);
    defer {
        const releaseEnvFn = api.*.ReleaseEnv orelse unreachable;
        releaseEnvFn(env);
    }
    
    // Create session options
    var session_opts: ?*onnx.OrtSessionOptions = null;
    const create_opts_fn = api.*.CreateSessionOptions orelse return OnnxError.UnwrapOnnxFnFoundNull;
    const opts_status = create_opts_fn(&session_opts);
    try onnx_core.checkStatus(opts_status, api);
    defer {
        const releaseSessionOptionsFn = api.*.ReleaseSessionOptions orelse unreachable;
        releaseSessionOptionsFn(session_opts);
    }
    
    // Create session with the test model
    const model_path = "test_data/with_past_kv.onnx";
    var maybe_session: ?*onnx.OrtSession = null;
    const createSessionFn = api.*.CreateSession orelse return OnnxError.UnwrapOnnxFnFoundNull;
    const session_status = createSessionFn(env, model_path.ptr, session_opts, &maybe_session);
    try onnx_core.checkStatus(session_status, api);
    if (maybe_session == null) return OnnxError.UnexpectedNull;
    const session: *onnx.OrtSession = maybe_session.?;
    defer {
        const releaseSessionFn = api.*.ReleaseSession orelse unreachable;
        releaseSessionFn(session);
    }
    
    // Parse model info
    const model_info = try parseModelInfo(allocator, onnx_allocator, api, session);
    defer model_info.destroy(allocator);
    
    // Validate that we have inputs and outputs
    try std.testing.expect(model_info.inputs.len > 0);
    try std.testing.expect(model_info.outputs.len > 0);
    
    // Check that we have some cached inputs (past_key_values)
    var has_cached_input = false;
    for (model_info.inputs) |input| {
        if (input.is_cached) {
            has_cached_input = true;
            // Verify that cached inputs have "past_key_values" in their name
            const input_name = std.mem.span(input.name);
            try std.testing.expect(std.mem.startsWith(u8, input_name, "past_key_values"));
        }
    }
    try std.testing.expect(has_cached_input);
    
    // Check that we have some cached outputs (past_key_values)
    var has_cached_output = false;
    for (model_info.outputs) |output| {
        if (output.is_cached) {
            has_cached_output = true;
            // Verify that cached outputs have "past_key_values" in their name
            const output_name = std.mem.span(output.name);
            try std.testing.expect(std.mem.startsWith(u8, output_name, "past_key_values"));
        }
    }
    try std.testing.expect(has_cached_output);
    
    // Verify all inputs have valid dimensions
    for (model_info.inputs) |input| {
        try std.testing.expect(input.dimensions.len > 0);
        // Check that dynamic dimensions (-1) have been replaced with 1
        for (input.dimensions) |dim| {
            try std.testing.expect(dim >= 1);
        }
    }
    
    // Verify all outputs have valid dimensions
    for (model_info.outputs) |output| {
        try std.testing.expect(output.dimensions.len > 0);
        // Check that dynamic dimensions (-1) have been replaced with 1
        for (output.dimensions) |dim| {
            try std.testing.expect(dim >= 1);
        }
    }
}


//CHECK FOR MEMORY LEAKS, SHOULD BE ALL OVER THE PLACE
//CREATE A SMALL WITH A TINY MODEL FIXTURE
