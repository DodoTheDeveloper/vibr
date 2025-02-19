const std = @import("std");

const ChatMessage = struct { role: []const u8, content: []const u8 };
const ChatResponse = struct { model: []const u8, created_at: []const u8, message: ChatMessage, done: bool };

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();
    const chatUri = try std.Uri.parse("http://127.0.0.1:11434/api/chat");
    var client = std.http.Client{ .allocator = allocator };
    defer client.deinit();
    var buf: [1024]u8 = undefined;

    const payload =
        \\{
        \\  "model": "hf.co/unsloth/DeepSeek-R1-Distill-Qwen-14B-GGUF:Q4_K_M",
        \\  "messages": [
        \\    { "role": "user", "content": "What is the following code doing? print(\"hello world\"" }
        \\  ],
        \\  "stream": true
        \\}
    ;
    var req = try client.open(.POST, chatUri, .{ .server_header_buffer = &buf });
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
        const parsed = try std.json.parseFromSlice(ChatResponse, allocator, line, .{ .ignore_unknown_fields = true });
        defer parsed.deinit();
        const msg: ChatResponse = parsed.value;
        std.debug.print("{s}", .{msg.message.content});
    }

    //const body = try rdr.readAllAlloc(allocator, 1024 * 1024 * 4);
    //defer allocator.free(body);
    std.debug.print("{s}", .{"Done reading"});
}

//fn printStreamResponse(rdr: *std.io.Reader) void {
//}

test "simple test" {
    var list = std.ArrayList(i32).init(std.testing.allocator);
    defer list.deinit(); // try commenting this out and see if zig detects the memory leak!
    try list.append(42);
    try std.testing.expectEqual(@as(i32, 42), list.pop());
}
