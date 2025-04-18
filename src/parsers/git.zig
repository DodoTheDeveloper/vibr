const std = @import("std");

pub const DiffRange = struct { start_line: usize, end_line: usize };

pub const FileGitDiff = struct {
    file_path: []const u8,
    diff_ranges: std.ArrayList(*DiffRange),

    pub fn new(allocator: std.mem.Allocator, file_path: []const u8) !*FileGitDiff {
        const _file_git_diff_ptr = try allocator.create(FileGitDiff);
        const _file_path = try allocator.alloc(u8, file_path.len);
        std.mem.copyForwards(u8, _file_path, file_path);
        _file_git_diff_ptr.*.file_path = _file_path;

        _file_git_diff_ptr.diff_ranges = std.ArrayList(*DiffRange).init(allocator);
        return _file_git_diff_ptr;
    }

    pub fn deinit(self: *FileGitDiff, allocator: std.mem.Allocator) void {
        allocator.free(self.*.file_path);
        self.*.diff_ranges.deinit();
    }
};

const DIFF_FILENAME_PREFIX = "diff --git a/";
const DIFF_LINE_CHANGE_PREFIX = "";
pub fn get_files_paths_from_git_diff(allocator: std.mem.Allocator, diff: []const u8) ![][]const u8 {
    // everything between diff --git a and a whitespace"
    var capture_list = std.ArrayList([]const u8).init(allocator);
    var line_iter = std.mem.tokenizeAny(u8, diff, "\n");

    while (line_iter.next()) |line| {
        if (std.mem.startsWith(u8, line, DIFF_FILENAME_PREFIX)) {
            var capured = line[DIFF_FILENAME_PREFIX.len..];
            const end_index = std.mem.indexOf(u8, capured, " ") orelse continue;
            capured = capured[0..end_index];
            try capture_list.append(capured);
        }
    }
    return try capture_list.toOwnedSlice();
}

pub fn get_file_git_diffs_from_unified_diff(allocator: std.mem.Allocator, diff: []const u8) !std.ArrayList(*FileGitDiff) {
    var file_git_diffs = std.ArrayList(*FileGitDiff).init(allocator);

    var line_iter = std.mem.tokenizeAny(u8, diff, "\n");

    while (line_iter.next()) |line| {
        std.debug.print("line {s}", .{line});

        if (std.mem.startsWith(u8, line, DIFF_FILENAME_PREFIX)) {
            var capured = line[DIFF_FILENAME_PREFIX.len..];
            const end_index = std.mem.indexOf(u8, capured, " ") orelse continue;
            const file_path = capured[0..end_index];

            const new_file_git_diff = try FileGitDiff.new(allocator, file_path);
            try file_git_diffs.append(new_file_git_diff);
        } else if (std.mem.startsWith(u8, line, DIFF_LINE_CHANGE_PREFIX)) {
            std.debug.print("hunk", .{});
            var tokenizer = std.mem.tokenize(u8, line, " @ -,+");
            _ = tokenizer.next(); // jumps over the hunks old start
            _ = tokenizer.next(); // jumps over the hunks old count
            const diff_start_slice = tokenizer.next().?;
            const diff_start = try std.fmt.parseInt(usize, diff_start_slice, 10);
            const diff_len_slice = tokenizer.next().?;
            const diff_len = try std.fmt.parseInt(usize, diff_len_slice, 10);

            std.debug.print("{s} {s}", .{ diff_start_slice, diff_len_slice });
            const new_range = try allocator.create(DiffRange);
            new_range.* = .{ .start_line = diff_start, .end_line = diff_start + diff_len };
            const current_diff = file_git_diffs.getLast();
            try current_diff.diff_ranges.append(new_range);
        }
    }

    return file_git_diffs;
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

test "Get FileGitDiff from git diff" {
    const given_allocator = std.testing.allocator;
    const file_path = "./test_data/git_diff_mock.txt";
    const file = try std.fs.cwd().openFile(file_path, .{});
    const given_diff = try file.readToEndAlloc(given_allocator, 2048);
    defer given_allocator.free(given_diff);
    const actual_file_git_diffs = try get_file_git_diffs_from_unified_diff(given_allocator, given_diff);

    try std.testing.expectEqual(2, actual_file_git_diffs.items.len);
    defer for (actual_file_git_diffs.items) |actual_file_git_diff| {
        actual_file_git_diff.*.deinit(given_allocator);
    };
}
