const std = @import("std");
const parsers_term = @import("parsers/term.zig");
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

    const program_args = parsers_term.parseArgs(&allocator, &args_list) catch |err| {
        std_err_writer.print("{}", .{err}) catch unreachable;
        return;
    };

    const file_content = utils.readFile(&allocator, program_args.filepath) catch |err| {
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
