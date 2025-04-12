const std = @import("std");
pub const ProgramArgs = struct { filepath: []const u8 };

pub const parseError = error{MissingFilepath};

pub fn parseArgs(allocator: std.mem.Allocator, args_list: *std.ArrayList([]const u8)) !*const ProgramArgs {
    const err_writer = std.io.getStdErr().writer();

    if (args_list.items.len > 2) {
        try err_writer.print("Missing file path. Length of program args {}", .{args_list.items.len});
    }

    const filepath: []const u8 = args_list.items[1];
    const programArgsPtr: *ProgramArgs = try allocator.create(ProgramArgs);
    programArgsPtr.* = ProgramArgs{ .filepath = filepath };

    return programArgsPtr;
}

test "parses args" {
    var given_allocator = std.testing.allocator;
    var given_list = std.ArrayList([]const u8).init(given_allocator);
    defer given_list.deinit();

    try given_list.append("program_name");
    try given_list.append("some/file/path.py");
    const actualArgsPtr: *const ProgramArgs = try parseArgs(given_allocator, &given_list);
    defer given_allocator.destroy(actualArgsPtr);

    const expected = ProgramArgs{ .filepath = "some/file/path.py" };
    try std.testing.expectEqual(expected, actualArgsPtr.*);
}
