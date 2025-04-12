const std = @import("std");

const DIFF_PREFIX = "diff --git a/";
pub fn get_files_paths_from_git_diff(allocator: std.mem.Allocator, diff: []const u8) ![][]const u8 {
    // everything between diff --git a and a whitespace"
    var capture_list = std.ArrayList([]const u8).init(allocator);
    var line_iter = std.mem.tokenizeAny(u8, diff, "\n");

    while (line_iter.next()) |line| {
        if (std.mem.startsWith(u8, line, DIFF_PREFIX)) {
            var capured = line[DIFF_PREFIX.len..];
            const end_index = std.mem.indexOf(u8, capured, " ") orelse continue;
            capured = capured[0..end_index];
            try capture_list.append(capured);
        }
    }
    return try capture_list.toOwnedSlice();
}

test "get_files_paths_from_git_diffgets filenames from git diff" {
    const given_allocator = std.testing.allocator;

    const file_path = "./test_data/git_diff_mock.txt";
    const file = try std.fs.cwd().openFile(file_path, .{});
    const given_diff = try file.readToEndAlloc(given_allocator, 2048);
    defer given_allocator.free(given_diff);

    const actual = try get_files_paths_from_git_diff(given_allocator, given_diff);
    defer given_allocator.free(actual);
    const expected = [_][]const u8{ "src/main.zig", "test_data/python/multiplication.py" };
    try std.testing.expectEqual(expected.len, actual.len);

    for (expected, 0..) |expected_slice, index| {
        const actual_slice = actual[index];
        try std.testing.expectEqualSlices(u8, expected_slice, actual_slice);
    }
}

test "get_files_paths_from_git_diffgets returns empty list when no diff" {
    const given_allocator = std.testing.allocator;
    const given_diff = "";

    const actual = try get_files_paths_from_git_diff(given_allocator, given_diff);
    defer given_allocator.free(actual);
    const expected = [_][]const u8{};
    try std.testing.expectEqual(expected.len, actual.len);

    for (expected, 0..) |expected_slice, index| {
        const actual_slice = actual[index];
        try std.testing.expectEqualSlices(u8, expected_slice, actual_slice);
    }
}
