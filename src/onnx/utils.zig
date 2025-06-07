const std = @import("std");
const onnx = @cImport({
    @cInclude("onnxruntime_c_api.h");
});
const ModelInfo = @import("./ModelInfo.zig").ModelInfo;
const onnx_core = @import("./core.zig");

pub const OnnxError = error{ UnwrapOnnxFnFoundNull, UnexpectedNull };

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
    std.debug.print("[DODO] input count {d}\n", .{num_inputs});

    const inputs: []ModelInfo.Input = try allocator.alloc(ModelInfo.Input, num_inputs);
    for (0..num_inputs) |input_idx| {
        var input_name_arr: [*:0]u8 = undefined;
        try onnx_core.SessionGetInputName(api, session, input_idx, onnx_allocator, &input_name_arr);

        var type_info: ?*onnx.OrtTypeInfo = null;
        try onnx_core.SessionGetInputTypeInfo(api, session, input_idx, &type_info);
        if (type_info == null) return OnnxError.UnexpectedNull;

        const input_name: []const u8 = std.mem.span(input_name_arr);
        const input_name_dup = try allocator.dupe(u8, input_name);

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

        std.debug.print("[DODO] input_name {s} dim_count {} dim_values {any} element_type: {any}\n", .{ input_name, dim_count, dim_values, element_type });
        const is_cached: bool = std.mem.startsWith(u8, input_name_dup, "past_key_values");

        inputs[input_idx] = .{ .name = input_name_dup, .data_type = element_type, .dimensions = dim_values, .is_cached = is_cached };
    }

    // outputs
    const sessionGetOutputCountFn = api.*.SessionGetOutputCount orelse return OnnxError.UnwrapOnnxFnFoundNull;
    var num_outputs: usize = 0;
    _ = sessionGetOutputCountFn(session, &num_outputs);
    std.debug.print("[DODO] output count {d}\n", .{num_inputs});

    const get_output_name_fn = api.*.SessionGetInputName orelse return OnnxError.UnwrapOnnxFnFoundNull;
    const outputs: []ModelInfo.Output = try allocator.alloc(ModelInfo.Output, num_outputs);

    for (0..num_outputs) |input_idx| {
        var output_name_arr: [*:0]u8 = undefined;
        try onnx_core.checkStatus(get_output_name_fn(session, input_idx, onnx_allocator, @ptrCast(&output_name_arr)), api);
        //const output_name: []const u8 = std.mem.span(output_name_arr);
        //const out_name_dup = try allocator.dupe(u8, output_name);

        //outputs[input_idx]  = .{.name=out_name_dup, .data_type=1, .dimensions=}
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
