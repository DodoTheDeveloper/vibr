const std = @import("std");
const concat = @import("../utils/slices.zig").concat;

pub const DeepSeekR1ResponseBuffer = struct {
    message: []const u8,
    pub fn new(allocator: std.mem.Allocator, message: []const u8) !*DeepSeekR1ResponseBuffer {
        const new_message = try allocator.create(DeepSeekR1ResponseBuffer);
        const _message = try allocator.alloc(u8, message.len);
        std.mem.copyForwards(u8, _message, message);
        new_message.* = .{ .message = _message };
        return new_message;
    }

    pub fn destroy(self: *DeepSeekR1ResponseBuffer, allocator: std.mem.Allocator) void {
        allocator.free(self.*.message);
        allocator.destroy(self);
    }

    pub fn appendMessage(self: *DeepSeekR1ResponseBuffer, allocator: std.mem.Allocator, to_append: []const u8) !void {
        const new_message = try concat(allocator, self.*.message, to_append);
        allocator.free(self.*.message);
        self.*.message = new_message;
    }

    pub fn getMessage(self: *DeepSeekR1ResponseBuffer) []const u8 {
        const start_pattern = "<think>";
        const end_pattern = "</think>\n";
        const think_start_pos = std.mem.indexOf(u8, self.*.message, start_pattern) orelse 0;
        const think_end_pos = std.mem.indexOf(u8, self.*.message, end_pattern) orelse 0;
        if (think_start_pos == 0 and think_end_pos == 0) {
            return self.*.message;
        }
        return self.*.message[think_end_pos + end_pattern.len ..];
    }
};

test "DeepSeekR1ResponseBuffer appends message" {
    const given_allocator = std.testing.allocator;
    const actual_message = try DeepSeekR1ResponseBuffer.new(given_allocator, "");

    defer actual_message.destroy(given_allocator);
    try std.testing.expectEqualSlices(u8, "", actual_message.*.message);
    try actual_message.appendMessage(given_allocator, "appended");
    try std.testing.expectEqualSlices(u8, "appended", actual_message.*.message);
    try actual_message.appendMessage(given_allocator, "appended");
    try std.testing.expectEqualSlices(u8, "appendedappended", actual_message.*.message);
}

test "DeepSeekR1ResponseBuffer.getMessage returns slice of message" {
    const given_allocator = std.testing.allocator;
    const given_buffer = try DeepSeekR1ResponseBuffer.new(given_allocator, "my_message");
    defer given_buffer.*.destroy(given_allocator);
    try std.testing.expectEqualSlices(u8, "my_message", given_buffer.*.getMessage());
}
test "DeepSeekR1ResponseBuffer.getMessage filters out '<think> .. </think>\n' pattern." {
    const given_allocator = std.testing.allocator;
    const given_buffer = try DeepSeekR1ResponseBuffer.new(given_allocator, "<think> thinking </think>\nmy_message");
    defer given_buffer.*.destroy(given_allocator);
    try std.testing.expectEqualSlices(u8, "my_message", given_buffer.*.getMessage());
}
