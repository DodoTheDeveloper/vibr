const std = @import("std");

pub fn escape_json_content(allocator: std.mem.Allocator, input: []const u8) ![]const u8 {
    var out = std.ArrayList(u8).init(allocator);
    // Loop through each character in the input.
    for (input) |c| {
        switch (c) {
            '\\' => try out.appendSlice("\\\\"), // Escape backslash.
            '\n' => try out.appendSlice("\\n"), // Escape newline.
            '\r' => try out.appendSlice("\\r"), // Escape carriage return.
            '\t' => try out.appendSlice("\\t"), // Escape tab.
            '"' => try out.appendSlice("\\\""), // Escape double quote.
            else => try out.append(c),
        }
    }
    return try out.toOwnedSlice();
}

pub fn readFile(allocator: std.mem.Allocator, file_path: []const u8) ![]const u8 {
    const HUNDRED_MEGABYTE = 1024 * 1000 * 100;
    const file = try std.fs.cwd().openFile(file_path, .{});
    defer file.close();
    const content = try file.readToEndAlloc(allocator, HUNDRED_MEGABYTE);
    return content;
}

test "escape json content" {
    const given_allocator = std.testing.allocator;

    // a\b"c' should be escaped as a\\b\"c\'
    const actual = try escape_json_content(given_allocator, "a\\b\"c");
    defer given_allocator.free(actual);
    try std.testing.expectEqualSlices(u8, "a\\\\b\\\"c", actual);
}

test "reads file" {
    const given_allocator = std.testing.allocator;

    const file_path = "./test_data/small.txt";
    const actual = try readFile(given_allocator, file_path);
    defer given_allocator.free(actual);
    const expected = "small\"\n";
    try std.testing.expectEqualSlices(u8, expected, actual);
}
