const std = @import("std");
const onnx = @cImport({
    @cInclude("onnxruntime_c_api.h");
});

const OnnxError = error{ FnDoesntExist, UnexpectedNull, ErrGettingApi };

pub fn checkStatus(status: ?*onnx.OrtStatus, api: *const onnx.OrtApi) !void {
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

pub fn OrtGetApiBase() !*const onnx.OrtApiBase {
    return onnx.OrtGetApiBase() orelse return OnnxError.UnexpectedNull;
}

pub fn GetApi(base: *const onnx.OrtApiBase) !*const onnx.OrtApi {
    const func = base.GetApi orelse return OnnxError.FnDoesntExist;
    return func(onnx.ORT_API_VERSION) orelse return OnnxError.UnexpectedNull;
}

pub fn GetAllocatorWithDefaultOptions(api: *const onnx.OrtApi, onnx_allocator: **onnx.OrtAllocator) !void {
    const func = api.*.GetAllocatorWithDefaultOptions orelse return OnnxError.FnDoesntExist;
    return try checkStatus(func(@ptrCast(onnx_allocator)), api);
}

pub fn SessionGetInputCount(api: *const onnx.OrtApi, session: *const onnx.OrtSession, out: *usize) !void {
    const func = api.*.SessionGetInputCount orelse return OnnxError.FnDoesntExist;
    return try checkStatus(func(session, out), api);
}

pub fn SessionGetInputName(api: *const onnx.OrtApi, session: *const onnx.OrtSession, index: usize, ort_allocator: *onnx.OrtAllocator, value: *[*:0]u8) !void {
    const func = api.*.SessionGetInputName orelse return OnnxError.FnDoesntExist;
    return try checkStatus(func(session, index, ort_allocator, @ptrCast(value)), api);
}

pub fn GetDimensionsCount(api: *const onnx.OrtApi, tensor_info: ?*const onnx.OrtTensorTypeAndShapeInfo, dim_count: *usize) !void {
    const func = api.*.GetDimensionsCount orelse return OnnxError.FnDoesntExist;
    return try checkStatus(func(tensor_info, dim_count), api);
}

pub fn GetDimensions(api: *const onnx.OrtApi, tensor_info: ?*const onnx.OrtTensorTypeAndShapeInfo, dim_values: [*]i64, dim_values_length: usize) !void {
    const func = api.*.GetDimensions orelse return OnnxError.FnDoesntExist;
    return try checkStatus(func(tensor_info, dim_values, dim_values_length), api);
}

pub fn SessionGetInputTypeInfo(api: *const onnx.OrtApi, session: *const onnx.OrtSession, idx: usize, type_info: *const ?*onnx.OrtTypeInfo) !void {
    const func = api.*.SessionGetInputTypeInfo orelse return OnnxError.FnDoesntExist;
    return try checkStatus(func(session, idx, @constCast(type_info)), api);
}

pub fn SessionGetOutputTypeInfo(api: *const onnx.OrtApi, session: *const onnx.OrtSession, idx: usize, type_info: *const ?*onnx.OrtTypeInfo) !void {
    const func = api.*.SessionGetOutputTypeInfo orelse return OnnxError.FnDoesntExist;
    return try checkStatus(func(session, idx, @constCast(type_info)), api);
}

pub fn CastTypeInfoToTensorInfo(api: *const onnx.OrtApi, type_info: ?*onnx.OrtTypeInfo, tensor_info: *const ?*const onnx.OrtTensorTypeAndShapeInfo) !void {
    const func = api.*.CastTypeInfoToTensorInfo orelse return OnnxError.FnDoesntExist;
    return try checkStatus(func(type_info, @constCast(tensor_info)), api);
}

pub fn GetTensorElementType(api: *const onnx.OrtApi, tensor_info: ?*const onnx.OrtTensorTypeAndShapeInfo, element_type: *onnx.ONNXTensorElementDataType) !void {
    const func = api.*.GetTensorElementType orelse return OnnxError.FnDoesntExist;
    return try checkStatus(func(tensor_info, element_type), api);
}

pub fn CreateTensorAsOrtValue(api: *const onnx.OrtApi, ort_allocator: *onnx.OrtAllocator, shape: *i64, shape_len: usize, data_type: onnx.ONNXTensorElementDataType, out: *const ?*onnx.OrtValue) !void {
    const func = api.*.CreateTensorAsOrtValue orelse return OnnxError.FnDoesntExist;
    try checkStatus(func(ort_allocator, shape, shape_len, data_type, @constCast(out)), api);
}

pub fn CreateTensorWithDataAsOrtValue(api: *const onnx.OrtApi, info: *const onnx.OrtMemoryInfo, data: *anyopaque, data_length: usize, shape: *i64, shape_len: usize, data_type: onnx.ONNXTensorElementDataType, out: *const ?*onnx.OrtValue) !void {
    const func = api.*.CreateTensorWithDataAsOrtValue orelse return OnnxError.FnDoesntExist;
    try checkStatus(func(info, data, data_length, shape, shape_len, data_type, @constCast(out)), api);
}
