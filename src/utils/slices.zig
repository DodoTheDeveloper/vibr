const std = @import("std");

/// Creates a new concatinated slice from a & b.
pub fn concat(allocator: std.mem.Allocator, a: []const u8, b: []const u8) ![]const u8 {
    const buffer = try allocator.alloc(u8, a.len + b.len);
    std.mem.copyForwards(u8, buffer[0..a.len], a);
    std.mem.copyForwards(u8, buffer[a.len..buffer.len], b);
    return buffer;
}

test "concatinates two []const u8 slices" {
    var given_allocator = std.testing.allocator;
    const actual_slice = try concat(given_allocator, "a", "b");
    defer given_allocator.free(actual_slice);
    try std.testing.expectEqualSlices(u8, "ab", actual_slice);
}
