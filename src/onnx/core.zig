const std = @import("std");
const ort = @cImport({
    @cInclude("onnxruntime_c_api.h");
});

const OnnxError = error{ FnDoesntExist, UnexpectedNull, ErrGettingApi };

pub fn checkStatus(status: ?*ort.OrtStatus, api: *const ort.OrtApi) !void {
    if (status) |st| {
        const getErrorMessageFn = api.*.GetErrorMessage orelse return OnnxError.FnDoesntExist;
        const msg = getErrorMessageFn(st);
        const stderr = std.io.getStdErr().writer();
        _ = try stderr.print("ONNX Runtime error: {s}\n", .{msg});
        std.debug.assert(api.ReleaseStatus != null);
        const releaseStatusFn = api.*.ReleaseStatus orelse return OnnxError.FnDoesntExist;
        releaseStatusFn(st);
        std.debug.panic("Abort due to ONNX runtime error\n", .{});
    }
}

pub fn OrtGetApiBase() !*const ort.OrtApiBase {
    return ort.OrtGetApiBase() orelse return OnnxError.UnexpectedNull;
}

pub fn GetApi(base: *const ort.OrtApiBase) !*const ort.OrtApi {
    const func = base.GetApi orelse return OnnxError.FnDoesntExist;
    return func(ort.ORT_API_VERSION) orelse return OnnxError.UnexpectedNull;
}

pub fn GetAllocatorWithDefaultOptions(api: *const ort.OrtApi, onnx_allocator: **ort.OrtAllocator) !void {
    const func = api.*.GetAllocatorWithDefaultOptions orelse return OnnxError.FnDoesntExist;
    return try checkStatus(func(@ptrCast(onnx_allocator)), api);
}

pub fn CreateEnv(api: *const ort.OrtApi, log_level: ort.OrtLoggingLevel, log_id: []const u8, out: **ort.OrtEnv) !void {
    var maybe_env_ptr: ?*ort.OrtEnv = null;
    const func = api.*.CreateEnv orelse return OnnxError.FnDoesntExist;
    try checkStatus(func(log_level, log_id.ptr, &maybe_env_ptr), api);
    out.* = maybe_env_ptr orelse return OnnxError.UnexpectedNull;
}

pub fn ReleaseEnv(api: *const ort.OrtApi, env: *ort.OrtEnv) !void {
    const func = api.*.ReleaseEnv orelse return OnnxError.FnDoesntExist;
    func(env);
}

pub fn CreateSessionOptions(api: *const ort.OrtApi, session_opts: **ort.OrtSessionOptions) !void {
    const func = api.*.CreateSessionOptions orelse return OnnxError.FnDoesntExist;
    try checkStatus(func(@ptrCast(session_opts)), api);
}

pub fn ReleaseSessionOptions(api: *const ort.OrtApi, session_opts: *ort.OrtSessionOptions) !void {
    const func = api.*.ReleaseSessionOptions orelse return OnnxError.FnDoesntExist;
    func(session_opts);
}

pub fn CreateSession(api: *const ort.OrtApi, env: *ort.OrtEnv, model_path: []const u8, session_opts: *ort.OrtSessionOptions, session_ptr: **ort.OrtSession) !void {
    var maybe_session_ptr: ?*ort.OrtSession = null;
    const func = api.*.CreateSession orelse return OnnxError.FnDoesntExist;
    try checkStatus(func(env, model_path.ptr, session_opts, @ptrCast(&maybe_session_ptr)), api);
    if (maybe_session_ptr == null) return OnnxError.UnexpectedNull;
    session_ptr.* = maybe_session_ptr.?;
}

pub fn CreateCpuMemoryInfo(api: *const ort.OrtApi, ort_allocator_type: ort.OrtAllocatorType, mem_type: ort.OrtMemType, mem_info: **ort.OrtMemoryInfo) !void {
    var maybe_mem_info: ?*ort.OrtMemoryInfo = null; const func = api.*.CreateCpuMemoryInfo orelse return OnnxError.FnDoesntExist;
    try checkStatus(func(ort_allocator_type, mem_type, @ptrCast(&maybe_mem_info)), api);
    if (maybe_mem_info == null) return OnnxError.UnexpectedNull;
    mem_info.* = maybe_mem_info.?;
}

pub fn SessionGetInputCount(api: *const ort.OrtApi, session: *const ort.OrtSession, out: *usize) !void {
    const func = api.*.SessionGetInputCount orelse return OnnxError.FnDoesntExist;
    try checkStatus(func(session, out), api);
}

pub fn SessionGetInputName(api: *const ort.OrtApi, session: *const ort.OrtSession, index: usize, ort_allocator: *ort.OrtAllocator, value: *[*:0]u8) !void {
    const func = api.*.SessionGetInputName orelse return OnnxError.FnDoesntExist;
    try checkStatus(func(session, index, ort_allocator, @ptrCast(value)), api);
}

pub fn GetDimensionsCount(api: *const ort.OrtApi, tensor_info: ?*const ort.OrtTensorTypeAndShapeInfo, dim_count: *usize) !void {
    const func = api.*.GetDimensionsCount orelse return OnnxError.FnDoesntExist;
    try checkStatus(func(tensor_info, dim_count), api);
}

pub fn GetDimensions(api: *const ort.OrtApi, tensor_info: ?*const ort.OrtTensorTypeAndShapeInfo, dim_values: [*]i64, dim_values_length: usize) !void {
    const func = api.*.GetDimensions orelse return OnnxError.FnDoesntExist;
    try checkStatus(func(tensor_info, dim_values, dim_values_length), api);
}

pub fn SessionGetInputTypeInfo(api: *const ort.OrtApi, session: *const ort.OrtSession, idx: usize, type_info: *const ?*ort.OrtTypeInfo) !void {
    const func = api.*.SessionGetInputTypeInfo orelse return OnnxError.FnDoesntExist;
    return try checkStatus(func(session, idx, @constCast(type_info)), api);
}

pub fn SessionGetOutputCount(api: *const ort.OrtApi, session: *const ort.OrtSession, out: *usize) !void {
    const func = api.*.SessionGetOutputCount orelse return OnnxError.FnDoesntExist;
    return try checkStatus(func(session, out), api);
}

pub fn SessionGetOutputName(api: *const ort.OrtApi, session: *const ort.OrtSession, index: usize, ort_allocator: *ort.OrtAllocator, value: *[*:0]u8) !void {
    const func = api.*.SessionGetOutputName orelse return OnnxError.FnDoesntExist;
    return try checkStatus(func(session, index, ort_allocator, @ptrCast(value)), api);
}

pub fn SessionGetOutputTypeInfo(api: *const ort.OrtApi, session: *const ort.OrtSession, idx: usize, type_info: *const ?*ort.OrtTypeInfo) !void {
    const func = api.*.SessionGetOutputTypeInfo orelse return OnnxError.FnDoesntExist;
    return try checkStatus(func(session, idx, @constCast(type_info)), api);
}

pub fn CastTypeInfoToTensorInfo(api: *const ort.OrtApi, type_info: ?*ort.OrtTypeInfo, tensor_info: *const ?*const ort.OrtTensorTypeAndShapeInfo) !void {
    const func = api.*.CastTypeInfoToTensorInfo orelse return OnnxError.FnDoesntExist;
    return try checkStatus(func(type_info, @constCast(tensor_info)), api);
}

pub fn GetTensorElementType(api: *const ort.OrtApi, tensor_info: ?*const ort.OrtTensorTypeAndShapeInfo, element_type: *ort.ONNXTensorElementDataType) !void {
    const func = api.*.GetTensorElementType orelse return OnnxError.FnDoesntExist;
    return try checkStatus(func(tensor_info, element_type), api);
}

pub fn CreateTensorAsOrtValue(api: *const ort.OrtApi, ort_allocator: *ort.OrtAllocator, shape: *i64, shape_len: usize, data_type: ort.ONNXTensorElementDataType, out: *const ?*ort.OrtValue) !void {
    const func = api.*.CreateTensorAsOrtValue orelse return OnnxError.FnDoesntExist;
    try checkStatus(func(ort_allocator, shape, shape_len, data_type, @constCast(out)), api);
}

pub fn CreateTensorWithDataAsOrtValue(api: *const ort.OrtApi, info: *const ort.OrtMemoryInfo, data: *anyopaque, data_length: usize, shape: *i64, shape_len: usize, data_type: ort.ONNXTensorElementDataType, out: *const ?*ort.OrtValue) !void {
    const func = api.*.CreateTensorWithDataAsOrtValue orelse return OnnxError.FnDoesntExist;
    try checkStatus(func(info, data, data_length, shape, shape_len, data_type, @constCast(out)), api);
}


pub fn Run(api: *const ort.OrtApi, session: *ort.OrtSession, run_options: ?*ort.OrtRunOptions, input_names:[][*c]const u8, input_values: []*ort.OrtValue, input_count: usize, output_names: [][*c]const u8,output_count: usize , output_values: []?*ort.OrtValue  ) !void {
    const func = api.*.Run orelse return OnnxError.FnDoesntExist;
    try checkStatus(func(session, run_options, @ptrCast(input_names.ptr), @ptrCast(input_values.ptr), input_count, @ptrCast(output_names.ptr), output_count, @ptrCast(output_values.ptr)), api);
}

