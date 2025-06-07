const std = @import("std");
const onnx = @cImport({
    @cInclude("onnxruntime_c_api.h");
});
const onnx_core = @import("./onnx/core.zig");
const onnx_utils = @import("./onnx/utils.zig");
const ModelInfo = @import("./onnx/ModelInfo.zig").ModelInfo;
const OrtValueBuffer = @import("./onnx/ModelInfo.zig").OrtValueBuffer;
const OnnxError = @import("./onnx/utils.zig").OnnxError;

pub fn run(allocator: std.mem.Allocator, model_path: [:0]const u8, prompt: []const i64) !void {
    std.debug.print("{d}\n", .{ prompt });

    const base: *const onnx.OrtApiBase = try onnx_core.OrtGetApiBase();
    const api: *const onnx.OrtApi = try onnx_core.GetApi(base);

    var onnx_allocator: *onnx.OrtAllocator = undefined;
    try onnx_core.GetAllocatorWithDefaultOptions(api, &onnx_allocator);

    var env: *onnx.OrtEnv = undefined;
    try onnx_core.CreateEnv(api, 
        onnx.ORT_LOGGING_LEVEL_WARNING,
        "zig_llm",
        &env, // out‐param
    );
    var session_opts: *onnx.OrtSessionOptions = undefined;
    try onnx_core.CreateSessionOptions(api, &session_opts);

    var session: *onnx.OrtSession = undefined;
    try onnx_core.CreateSession(api,env, model_path, session_opts, &session);

    // collecting infos about the models inputs
    const model_info = try onnx_utils.parseModelInfo(allocator, onnx_allocator, api, session);

    var mem_info: *onnx.OrtMemoryInfo = undefined;
    try onnx_core.CreateCpuMemoryInfo(api, onnx.OrtArenaAllocator, onnx.OrtMemTypeDefault, &mem_info);

    var input_value_ptrs = try allocator.alloc(*onnx.OrtValue, model_info.inputs.len);
    const seq_len: i64 = 5;
    //
    const input_ids: OrtValueBuffer = .{ .I64 = &.{ 1234, 5678, 9012, 3456, 7890 } };
    const attention_mask: OrtValueBuffer = .{ .I64 = &.{ 1, 1, 1, 1, 1 } };
    const position_ids: OrtValueBuffer = .{ .I64 = &.{ 0, 1, 2, 3, 4 } };
    const shape: [2]i64 = .{ 1, seq_len };

    var input_tensor_ids: ?*onnx.OrtValue = null;
    try onnx_core.CreateTensorWithDataAsOrtValue(api, mem_info, @constCast(@ptrCast(input_ids.I64.ptr)), input_ids.I64.len * @sizeOf(i64), @constCast(&shape[0]), shape.len, onnx.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &input_tensor_ids);
    if (input_tensor_ids == null) return OnnxError.UnexpectedNull;
    input_value_ptrs[0] = input_tensor_ids.?;

    var input_tensor_mask: ?*onnx.OrtValue = null;
    try onnx_core.CreateTensorWithDataAsOrtValue(api, mem_info, @constCast(@ptrCast(attention_mask.I64.ptr)), attention_mask.I64.len * @sizeOf(i64), @constCast(&shape[0]), shape.len, onnx.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &input_tensor_mask);
    if (input_tensor_mask == null) return OnnxError.UnexpectedNull;
    input_value_ptrs[1] = input_tensor_mask.?;

    var input_tensor_pos: ?*onnx.OrtValue = null;
    try onnx_core.CreateTensorWithDataAsOrtValue(api, mem_info, @constCast(@ptrCast(position_ids.I64.ptr)), position_ids.I64.len * @sizeOf(i64), @constCast(&shape[0]), shape.len, onnx.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &input_tensor_pos);
    if (input_tensor_pos == null) return OnnxError.UnexpectedNull;
    input_value_ptrs[2] = input_tensor_pos.?;

    var input_name_ptrs = try allocator.alloc([*c]const u8, model_info.inputs.len);
    defer allocator.free(input_name_ptrs);
    var output_name_ptrs = try allocator.alloc([*c]const u8, model_info.outputs.len);
    defer allocator.free(output_name_ptrs);

    var input_buffers = try allocator.alloc(OrtValueBuffer, model_info.inputs.len);
    input_buffers[0] = input_ids;
    input_buffers[1] = attention_mask;
    input_buffers[2] = position_ids;

    // create cached inputs
    for (model_info.*.inputs, 0..) |input, idx| {
        // fill input_buffer with buffers
        input_name_ptrs[idx] = &input.name[0];
    


        if (!input.is_cached) continue;
        const total_dim_product = onnx_utils.getTensorDimProduct(input.dimensions);
        var ort_value: ?*onnx.OrtValue = null;
        switch (input.data_type) {
            onnx.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT => {
                var buffer = try allocator.alloc(f32, total_dim_product);
                @memset(buffer, 0);
                try onnx_core.CreateTensorWithDataAsOrtValue(api, mem_info, @ptrCast(&buffer[0]), total_dim_product * @sizeOf(f32), &input.dimensions[0], input.dimensions.len, onnx.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &ort_value);
                if (ort_value == null) return OnnxError.UnexpectedNull;
                input_value_ptrs[idx] = ort_value.?;
            },
            onnx.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64 => {
                var buffer = try allocator.alloc(i64, total_dim_product);
                @memset(buffer, 0);
                try onnx_core.CreateTensorWithDataAsOrtValue(api, mem_info, @ptrCast(&buffer[0]), total_dim_product * @sizeOf(i64), &input.dimensions[0], input.dimensions.len, onnx.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &ort_value);
                if (ort_value == null) return OnnxError.UnexpectedNull;
                input_value_ptrs[idx] = ort_value.?;
            },
            else => return OnnxError.UnsupportedTensorDataType
        }
    }

    //var output_buffers = try allocator.alloc([]OrtValueBuffer, model_info.inputs.len );
    // create cached inputs
    var output_tensors = try allocator.alloc(?*onnx.OrtValue, model_info.*.outputs.len);
    for (model_info.*.outputs, 0..) |output, idx| {
        output_name_ptrs[idx] = &output.name[0];
        output_tensors[idx] = null;
    }
    
    //const runFn = api.*.Run orelse return OnnxError.UnwrapOnnxFnFoundNull;
    //const run_status = runFn(session, null, @ptrCast(input_name_ptrs.ptr), @ptrCast(input_value_ptrs.ptr), model_info.inputs.len, @ptrCast(output_name_ptrs.ptr), model_info.outputs.len, @ptrCast(output_tensors.ptr));
    try onnx_core.Run(api, session, null, input_name_ptrs, input_value_ptrs, model_info.inputs.len, output_name_ptrs, model_info.outputs.len, output_tensors);
    std.debug.print("[DODO] inference ran.\n", .{});
    //try checkStatus(run_status, api);


    //const getTensorMutableDataFn = api.*.GetTensorMutableData orelse return OnnxError.UnwrapOnnxFnFoundNull;
    //_ = getTensorMutableDataFn(output_tensors, &raw_data);

    //const data_ptr = raw_data.?;
    //const stdout_writer = std.io.getStdOut().writer();
    //_ = try stdout_writer.print("data_ptr {any}", .{data_ptr});

    // ======================

    const releaseValueFn = api.*.ReleaseValue orelse return OnnxError.UnwrapOnnxFnFoundNull;
    releaseValueFn(input_tensor_ids);
    releaseValueFn(input_tensor_mask);
    releaseValueFn(input_tensor_pos);
    //releaseValueFn(output_tensors);

    const releaseMemoryInfoFn = api.*.ReleaseMemoryInfo orelse return OnnxError.UnwrapOnnxFnFoundNull;
    releaseMemoryInfoFn(mem_info);

    const releaseSessionFn = api.*.ReleaseSession orelse return OnnxError.UnwrapOnnxFnFoundNull;
    releaseSessionFn(session);
    const releaseSessionOptionsFn = api.*.ReleaseSessionOptions orelse return OnnxError.UnwrapOnnxFnFoundNull;
    releaseSessionOptionsFn(session_opts);
    const releaseEnvFn = api.*.ReleaseEnv orelse return OnnxError.UnwrapOnnxFnFoundNull;
    releaseEnvFn(env);
}
