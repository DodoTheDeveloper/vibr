const std = @import("std");

const ChatMessage = struct { role: []const u8, content: []const u8 };
const ChatResponse = struct { model: []const u8, created_at: []const u8, message: ChatMessage, done: bool };

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    var allocator = gpa.allocator();
    const stdout = std.io.getStdOut();
    const stdErrWriter = std.io.getStdErr().writer();
    _ = stdout.write("Input:\n") catch unreachable;
    var userInputBuffer: [1024_000]u8 = undefined;

    var reader = std.io.getStdIn().reader();
    const maybeInput = readUserInput(userInputBuffer[0..], &reader) catch |err| {
        try stdErrWriter.print("An error occured while parsing input: {}", .{err});
        return;
    };
    const input = maybeInput orelse {
        try stdErrWriter.print("No valid input", .{});
        return;
    };

    const payload = createPayload(&allocator, input) catch |err| {
        try stdErrWriter.print("An error occured while making the request: {}", .{err});
        return;
    };
    defer allocator.free(payload);

    std.debug.print("3 {s}", .{payload});
    makeRequest(&allocator, payload) catch |err| {
        try stdErrWriter.print("An error occured while making the request: {}", .{err});
        return;
    };
}

/// Creates the payload for a request.
fn createPayload(allocator: *std.mem.Allocator, content: []const u8) ![]u8 {
    const template =
        \\{{
        \\  "model": "hf.co/unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF:Q4_K_M",
        \\  "messages": [
        \\    {{ "role": "user", "content": "{s}" }}
        \\  ],
        \\  "stream": true
        \\}}
    ;
    return std.fmt.allocPrint(allocator.*, template, .{content});
}

fn makeRequest(allocator: *std.mem.Allocator, payload: []const u8) !void {
    const chatUri = try std.Uri.parse("http://127.0.0.1:11434/api/chat");
    var client = std.http.Client{ .allocator = allocator.* };
    defer client.deinit();

    var headerBuff: [1024]u8 = undefined;
    var req = try client.open(.POST, chatUri, .{ .server_header_buffer = &headerBuff });
    defer req.deinit();
    req.transfer_encoding = .{ .content_length = payload.len };
    try req.send();
    var wtr = req.writer();
    try wtr.writeAll(payload);
    try req.finish();
    try req.wait();
    var rdr = req.reader();
    var readBuf: [2048]u8 = undefined;

    while (try rdr.readUntilDelimiterOrEof(&readBuf, '\n')) |line| {
        //std.debug.print("resp: {s}", .{line});
        const parsed = try std.json.parseFromSlice(ChatResponse, allocator.*, line, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();
        const msg: ChatResponse = parsed.value;
        std.debug.print("{s}", .{msg.message.content});
    }
}

/// Reads & formats and writes the user input into the provided `buffer`.
fn readUserInput(buffer: []u8, reader: anytype) !?[]u8 {
    const userInput = try reader.readUntilDelimiterOrEof(buffer, '\n');
    if (userInput) |input| {
        return if (input[input.len - 1] == '\n') input[0 .. input.len - 1] else input[0..];
    }
    return userInput;
}

test "simple test" {
    var list = std.ArrayList(i32).init(std.testing.allocator);
    defer list.deinit(); // try commenting this out and see if zig detects the memory leak!
    try list.append(42);
    try std.testing.expectEqual(@as(i32, 42), list.pop());
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

test "creates payload" {
    const expected_payload =
        \\{
        \\  "model": "hf.co/unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF:Q4_K_M",
        \\  "messages": [
        \\    { "role": "user", "content": "my prompt" }
        \\  ],
        \\  "stream": true
        \\}
    ;
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    var allocator = gpa.allocator();

    const payload = createPayload(&allocator, "my prompt"[0..]) catch unreachable;
    defer allocator.free(payload);
    try std.testing.expect(std.mem.eql(u8, payload, expected_payload));
}
