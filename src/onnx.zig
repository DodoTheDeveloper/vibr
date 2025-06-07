const std = @import("std");
const onnx = @cImport({
    @cInclude("onnxruntime_c_api.h");
});
const onnx_core = @import("./onnx/core.zig");
const onnx_utils = @import("./onnx/utils.zig");
const ModelInfo = @import("./onnx/ModelInfo.zig").ModelInfo;
const OnnxError = @import("./onnx/utils.zig").OnnxError;

fn checkStatus(status: ?*onnx.OrtStatus, api: *const onnx.OrtApi) !void {
    if (status) |st| {
        const getErrorMessageFn = api.*.GetErrorMessage orelse return OnnxError.UnwrapOnnxFnFoundNull;
        const msg = getErrorMessageFn(st);
        const stderr = std.io.getStdErr().writer();
        _ = try stderr.print("ONNX Runtime error: {s}\n", .{msg});
        std.debug.assert(api.ReleaseStatus != null);
        const releaseStatusFn = api.*.ReleaseStatus orelse return OnnxError.UnwrapOnnxFnFoundNull;
        releaseStatusFn(st);
        std.debug.panic("Abort due to ONNX runtime error\n", .{});
    }
}

pub fn run(allocator: std.mem.Allocator, model_path: [:0]const u8, prompt: []const i64) !void {
    std.debug.print("{}{d}\n", .{ allocator, prompt });

    const base: *const onnx.OrtApiBase = try onnx_core.OrtGetApiBase();
    const api: *const onnx.OrtApi = try onnx_core.GetApi(base);

    var onnx_allocator: *onnx.OrtAllocator = undefined;
    try onnx_core.GetAllocatorWithDefaultOptions(api, &onnx_allocator);

    var maybe_env_ptr: ?*onnx.OrtEnv = null;
    const create_env_fn = api.*.CreateEnv orelse return OnnxError.UnwrapOnnxFnFoundNull;

    const status = create_env_fn(
        onnx.ORT_LOGGING_LEVEL_WARNING,
        "zig_llm",
        &maybe_env_ptr, // out‐param
    );
    const env = maybe_env_ptr orelse return OnnxError.UnwrapOnnxFnFoundNull;

    if (status) |errPtr| {
        // error …
        const get_error_message_fn = api.*.GetErrorMessage orelse return OnnxError.UnwrapOnnxFnFoundNull;
        std.debug.print("ORT failed: {s}\n", .{get_error_message_fn(errPtr)});
    } else {
        std.debug.print("ORT env created!\n", .{});
    }

    var session_opts: ?*onnx.OrtSessionOptions = null;
    const create_opts_fn = api.*.CreateSessionOptions orelse return OnnxError.UnwrapOnnxFnFoundNull;
    _ = create_opts_fn(&session_opts);

    var maybe_session: ?*onnx.OrtSession = null;
    const createSessionFn = api.*.CreateSession orelse return OnnxError.UnwrapOnnxFnFoundNull;
    _ = createSessionFn(env, model_path.ptr, session_opts, &maybe_session);
    if (maybe_session == null) return OnnxError.UnexpectedNull;
    const session: *onnx.OrtSession = maybe_session.?;

    // collecting infos about the models inputs
    const model_info = try onnx_utils.parseModelInfo(allocator, onnx_allocator, api, session);

    var mem_info: *onnx.OrtMemoryInfo = undefined;
    const create_cpu_Memory_info_Fn = api.*.CreateCpuMemoryInfo orelse return OnnxError.UnwrapOnnxFnFoundNull;
    try checkStatus(create_cpu_Memory_info_Fn(onnx.OrtArenaAllocator, onnx.OrtMemTypeDefault, @ptrCast(&mem_info)), api);
    // ==implement run here==

    const seq_len: i64 = 5;
    var input_ids: [5]i64 = .{ 1234, 5678, 9012, 3456, 7890 };
    var attention_mask: [5]f32 = .{ 1, 1, 1, 1, 1 };
    var position_ids: [5]i64 = .{ 0, 1, 2, 3, 4 };
    var shape: [2]i64 = .{ 1, seq_len };

    var input_tensor_ids: ?*onnx.OrtValue = null;
    try onnx_core.CreateTensorWithDataAsOrtValue(api, mem_info, @ptrCast(&input_ids[0]), input_ids.len * @sizeOf(i64), &shape[0], shape.len, onnx.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &input_tensor_ids);

    var input_tensor_mask: ?*onnx.OrtValue = null;
    try onnx_core.CreateTensorWithDataAsOrtValue(api, mem_info, @ptrCast(&attention_mask[0]), attention_mask.len * @sizeOf(i64), &shape[0], shape.len, onnx.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &input_tensor_mask);

    var input_tensor_pos: ?*onnx.OrtValue = null;
    try onnx_core.CreateTensorWithDataAsOrtValue(api, mem_info, @ptrCast(&position_ids[0]), position_ids.len * @sizeOf(i64), &shape[0], shape.len, onnx.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &input_tensor_pos);

    var cache_tensors = std.ArrayList(*const ?*onnx.OrtValue).init(allocator);
    defer cache_tensors.deinit();
    var cache_tensor_buffers = std.ArrayList([]f32).init(allocator);
    defer cache_tensor_buffers.deinit();

    std.debug.print("[DODO] HERE\n", .{});
    // create cached inputs
    for (model_info.*.inputs) |input| {
        if (!input.is_cached) continue;
        const total_dim_product = onnx_utils.getTensorDimProduct(input.dimensions);
        var buffer = try allocator.alloc(f32, total_dim_product);
        @memset(buffer, 0);
        var ort_value: ?*onnx.OrtValue = null;
        std.debug.print("[DODO] HERE 1 {any}, {any} {any}\n", .{ input, total_dim_product, @sizeOf(f32) });

        try onnx_core.CreateTensorWithDataAsOrtValue(api, mem_info, @ptrCast(&buffer[0]), total_dim_product * @sizeOf(f32), &input.dimensions[0], input.dimensions.len, onnx.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &ort_value);
        std.debug.print("[DODO] HERE 2\n", .{});
        try cache_tensors.append(&ort_value);
        try cache_tensor_buffers.append(buffer);
    }

    const input_ids_name: [*:0]const u8 = "input_ids";
    const attention_mask_name: [*:0]const u8 = "attention_mask";
    const position_ids_name: [*:0]const u8 = "position_ids";
    const in_names_c = [_][*:0]const u8{ input_ids_name, attention_mask_name, position_ids_name };

    const logits_name: [*:0]const u8 = "logits";
    const out_names_c = [_][*:0]const u8{logits_name};

    var output_tensors: ?*onnx.OrtValue = null;
    const inputs = [_]?*onnx.OrtValue{ input_tensor_ids, input_tensor_mask, input_tensor_pos };

    std.debug.print("[DODO] input_tensor_mask {any}", .{input_tensor_mask});
    const runFn = api.*.Run orelse return OnnxError.UnwrapOnnxFnFoundNull;
    const run_status = runFn(session, null, &in_names_c[0], &inputs, in_names_c.len, &out_names_c[0], out_names_c.len, &output_tensors);
    try checkStatus(run_status, api);

    var raw_data: ?*anyopaque = null;

    const getTensorMutableDataFn = api.*.GetTensorMutableData orelse return OnnxError.UnwrapOnnxFnFoundNull;
    _ = getTensorMutableDataFn(output_tensors, &raw_data);

    const data_ptr = raw_data.?;
    const stdout_writer = std.io.getStdOut().writer();
    _ = try stdout_writer.print("data_ptr {any}", .{data_ptr});

    // ======================

    const releaseValueFn = api.*.ReleaseValue orelse return OnnxError.UnwrapOnnxFnFoundNull;
    releaseValueFn(input_tensor_ids);
    releaseValueFn(input_tensor_mask);
    releaseValueFn(input_tensor_pos);
    releaseValueFn(output_tensors);

    const releaseMemoryInfoFn = api.*.ReleaseMemoryInfo orelse return OnnxError.UnwrapOnnxFnFoundNull;
    releaseMemoryInfoFn(mem_info);

    const releaseSessionFn = api.*.ReleaseSession orelse return OnnxError.UnwrapOnnxFnFoundNull;
    releaseSessionFn(session);
    const releaseSessionOptionsFn = api.*.ReleaseSessionOptions orelse return OnnxError.UnwrapOnnxFnFoundNull;
    releaseSessionOptionsFn(session_opts);
    const releaseEnvFn = api.*.ReleaseEnv orelse return OnnxError.UnwrapOnnxFnFoundNull;
    releaseEnvFn(env);
}
