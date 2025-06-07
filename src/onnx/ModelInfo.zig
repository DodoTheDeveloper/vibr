const std = @import("std");
const onnx = @cImport({
    @cInclude("onnxruntime_c_api.h");
});


pub const OrtValueBuffer = union(enum) {
    F32: []const f32,
    I64: []const i64
};

pub const ModelInfo = struct {
    pub const Input = struct { name: [*:0]const u8, data_type: onnx.ONNXTensorElementDataType, dimensions: []i64, is_cached: bool };
    pub const Output = struct { name: [*:0]const u8, data_type: onnx.ONNXTensorElementDataType, dimensions: []i64, is_cached: bool };

    inputs: []Input,
    outputs: []Output,

    pub fn create(allocator: std.mem.Allocator, inputs: []Input, outputs: []Output) !*ModelInfo {
        const model_info = try allocator.create(ModelInfo);
        model_info.* = .{ .inputs = inputs, .outputs = outputs };
        return model_info;
    }

    pub fn destroy(self: *ModelInfo, allocator: std.mem.Allocator) void {
        for (self.inputs) |input| {
            allocator.free(std.mem.span(input.name));
            allocator.free(input.dimensions);
        }
        allocator.free(self.inputs);
        for (self.outputs) |output| {
            allocator.free(std.mem.span(output.name));
            allocator.free(output.dimensions);
        }
        allocator.free(self.outputs);
        allocator.destroy(self);
    }
};

test "ModelInfo create & destroy" {
    const given_allocator = std.testing.allocator;
    const given_inputs = try given_allocator.alloc(ModelInfo.Input, 2);
    given_inputs[0] = .{ .name = try given_allocator.dupeZ(u8, "input_0"), .data_type = 0, .dimensions = try given_allocator.dupe(i64, &.{ 1, 2, 3 }), .is_cached = false };
    given_inputs[1] = .{ .name = try given_allocator.dupeZ(u8, "past_key_values_1"), .data_type = 0, .dimensions = try given_allocator.dupe(i64, &.{ 1, 2, 3 }), .is_cached = true };

    const given_outputs = try given_allocator.alloc(ModelInfo.Output, 2);
    given_outputs[0] = .{ .name = try given_allocator.dupeZ(u8, "output_0"), .data_type = onnx.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, .dimensions = try given_allocator.dupe(i64, &.{ 1, 2, 3 }), .is_cached = false };
    given_outputs[1] = .{ .name = try given_allocator.dupeZ(u8, "past_key_values_1"), .data_type = onnx.ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, .dimensions = try given_allocator.dupe(i64, &.{ 1, 2, 3 }), .is_cached = true };

    const actual_model_info = try ModelInfo.create(given_allocator, given_inputs, given_outputs);
    defer actual_model_info.*.destroy(given_allocator);

    //try std.testing.expectEqualSlices(ModelInfo.Input, given_inputs, actual_model_info.inputs);
    //try std.testing.expectEqualSlices(ModelInfo.Output, given_outputs, actual_model_info.outputs);
}
