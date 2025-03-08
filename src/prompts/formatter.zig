const std = @import("std");

const prompt = "For the following Python code add docstrings to all functions missing them. You should only respond with the modified code. No explenation except inside of the code needed. The Python code is as follows: ";
pub fn format_code_comment_promt(allocator: *std.mem.Allocator, file_content: []const u8) ![]const u8 {
    const combined_len = file_content.len + prompt.len;
    var concat = try allocator.alloc(u8, combined_len);
    std.mem.copyForwards(u8, concat[0..prompt.len], prompt);
    std.mem.copyForwards(u8, concat[prompt.len..combined_len], file_content);
    return concat;
}

test "format_code_comment_promt formats prompt" {
    var given_gpa = std.heap.GeneralPurposeAllocator(.{}){};
    var given_allocator = given_gpa.allocator();

    const given_file_content = "some content";
    const actual = try format_code_comment_promt(&given_allocator, given_file_content);
    const expected = "For the following Python code add docstrings to all functions missing them. You should only respond with the modified code. No explenation except inside of the code needed. The Python code is as follows: some content";
    try std.testing.expectEqualSlices(u8, expected, actual);
}
