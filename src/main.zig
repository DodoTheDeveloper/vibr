const std = @import("std");
const parsers_term = @import("parsers/term.zig");
const parsers_git = @import("parsers/git.zig");
const utils = @import("utils.zig");
const prompts_formatter = @import("prompts/formatter.zig");
const requests = @import("requests.zig");

pub fn main() void {
    var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
    defer arena.deinit();
    var allocator = arena.allocator();

    const std_err_writer = std.io.getStdErr().writer();

    // get args
    var args_iter = try std.process.argsWithAllocator(allocator);
    defer args_iter.deinit();
    var args_list = std.ArrayList([]const u8).init(allocator);
    defer args_list.deinit();

    while (true) {
        const maybe_arg = args_iter.next();
        if (maybe_arg == null) break;
        args_list.append(maybe_arg.?) catch |err| {
            std_err_writer.print("{}", .{err}) catch unreachable;
        };
    }

    //const program_args = parsers_term.parseArgs(&allocator, &args_list) catch |err| {
    //    std_err_writer.print("{}", .{err}) catch unreachable;
    //    return;
    //};

    // get git diff
    const git_diff = run_git_diff_main(allocator) catch |err| {
        std_err_writer.print("An error occured while running 'git diff': {}", .{err}) catch unreachable;
        return;
    };
    defer allocator.free(git_diff);
    // get file paths from git diff
    const file_paths = parsers_git.get_files_paths_from_git_diff(allocator, git_diff) catch |err| {
        std_err_writer.print("An error occured while parsing the git diff: {}", .{err}) catch unreachable;
        return;
    };

    for (file_paths) |file_path| {
        const file_content = utils.readFile(&allocator, file_path) catch |err| {
            std_err_writer.print("{}", .{err}) catch unreachable;
            return;
        };
        defer allocator.free(file_content);

        const formatted_prompt = prompts_formatter.format_code_comment_promt(&allocator, file_content) catch |err| {
            std_err_writer.print("{}", .{err}) catch unreachable;
            return;
        };
        defer allocator.free(formatted_prompt);

        requests.send_request_to_ollama(&allocator, formatted_prompt) catch |err| {
            std_err_writer.print("An error occured while making the request: {}", .{err}) catch unreachable;
            return;
        };
    }
}

fn run_git_diff_main(allocator: std.mem.Allocator) ![]u8 {
    // Initialize the child process with the command and its arguments.
    const args: []const []const u8 = &[_][]const u8{ "git", "diff", "main" };
    const result = try std.process.Child.run(.{ .allocator = allocator, .argv = args });

    const output: []u8 = try allocator.alloc(u8, result.stdout.len);
    const stdout = result.stdout;
    std.mem.copyForwards(u8, output, stdout);
    return output;
}

/// Reads & formats and writes the user input into the provided `buffer`.
fn readUserInput(buffer: []u8, reader: anytype) !?[]u8 {
    const userInput = try reader.readUntilDelimiterOrEof(buffer, '\n');
    if (userInput) |input| {
        if (input.len == 0) return userInput;
        return if (input[input.len - 1] == '\n') input[0 .. input.len - 1] else input[0..];
    }
    return userInput;
}

test "read user input" {
    var given_buffer: [1024]u8 = undefined;
    const expected_input = "user input";
    var given_stream = std.io.fixedBufferStream(expected_input);
    const input = try readUserInput(given_buffer[0..], given_stream.reader());
    try std.testing.expect(std.mem.eql(u8, input.?, expected_input));
}

test "read user input trailing \\n" {
    var given_buffer: [1024]u8 = undefined;
    const expected_input = "user input";
    var given_stream = std.io.fixedBufferStream(expected_input ++ "\n");
    const input = try readUserInput(given_buffer[0..], given_stream.reader());
    try std.testing.expect(std.mem.eql(u8, input.?, expected_input));
}
