const std = @import("std");
const utils = @import("utils.zig");
const DeepSeekR1ResponseBuffer = @import("models/DeepSeekR1ResponseBuffer.zig").DeepSeekR1ResponseBuffer;

pub const ChatMessage = struct { role: []const u8, content: []const u8 };
pub const ChatResponse = struct { model: []const u8, created_at: []const u8, message: ChatMessage, done: bool };

pub fn send_request_to_ollama(allocator: std.mem.Allocator, prompt: []const u8) !*DeepSeekR1ResponseBuffer {
    const payload = try create_payload(allocator, prompt);
    return try make_request(allocator, payload);
}

fn make_request(allocator: std.mem.Allocator, payload: []const u8) !*DeepSeekR1ResponseBuffer {
    const chatUri = try std.Uri.parse("http://127.0.0.1:11434/api/chat");
    var client = std.http.Client{ .allocator = allocator };
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
    const requestBuffer = try DeepSeekR1ResponseBuffer.new(allocator, "");

    while (try rdr.readUntilDelimiterOrEof(&readBuf, '\n')) |line| {
        const parsed = try std.json.parseFromSlice(ChatResponse, allocator, line, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();

        try requestBuffer.*.appendMessage(allocator, parsed.value.message.content);
    }
    return requestBuffer;
}

/// Creates the payload for a request.
fn create_payload(allocator: std.mem.Allocator, content: []const u8) ![]u8 {
    const escaped_contet = try utils.escape_json_content(allocator, content);
    defer allocator.free(escaped_contet);
    const template =
        \\{{
        \\  "model": "hf.co/unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF:Q4_K_M",
        \\  "messages": [
        \\    {{ "role": "user", "content": "{s}" }}
        \\  ],
        \\  "stream": true
        \\}}
    ;

    return std.fmt.allocPrint(allocator, template, .{escaped_contet});
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
    var allocator = std.testing.allocator;

    const payload = create_payload(allocator, "my prompt"[0..]) catch unreachable;
    defer allocator.free(payload);
    try std.testing.expect(std.mem.eql(u8, payload, expected_payload));
}
