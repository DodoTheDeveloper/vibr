const std = @import("std");

pub const Tokenizer = struct {
    vocab_path: []const u8,
    merges_path: []const u8,
    vocab: std.StringHashMap(u32),
    merges: std.StringHashMap(std.StringHashMap(usize)),

    pub fn init(allocator: std.mem.Allocator, vocab_path: []const u8, merges_path: []const u8) !*Tokenizer {
        const tokenizer_ptr = try allocator.create(Tokenizer);
        const _vocab_path = try allocator.alloc(u8, vocab_path.len);
        std.mem.copyForwards(u8, _vocab_path, vocab_path);
        const _merges_path = try allocator.alloc(u8, merges_path.len);
        std.mem.copyForwards(u8, _merges_path, merges_path);
        tokenizer_ptr.* = .{ .vocab_path = _vocab_path, .merges_path = _merges_path, .vocab = std.StringHashMap(u32).init(allocator), .merges = std.StringHashMap(std.StringHashMap(usize)).init(allocator) };

        return tokenizer_ptr;
    }
    pub fn destroy(self: *Tokenizer, allocator: std.mem.Allocator) void {
        std.debug.print("Start Deinit Tokenizer\n", .{});
        allocator.free(self.*.vocab_path);
        allocator.free(self.*.merges_path);

        std.debug.print("Deinit vocab\n", .{});
        self.*.vocab.deinit();
        std.debug.print("Finished deinit vocab\n", .{});

        std.debug.print("Deinit merges\n", .{});
        var it = self.merges.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.*.deinit();
        }

        self.*.merges.deinit();
        std.debug.print("Finished deinit merges\n", .{});
        allocator.destroy(self);

        std.debug.print("Finished deinit Tokenizer\n", .{});
    }

    fn loadVocab(self: *Tokenizer, allocator: std.mem.Allocator) !void {
        std.debug.print("Start loadVocab\n", .{});
        const TEN_MEGABYTE = 10 * 1024 * 1024;
        const vocab_bytes = try std.fs.cwd().readFileAlloc(allocator, self.*.vocab_path, TEN_MEGABYTE);
        defer allocator.free(vocab_bytes);
        const json = try std.json.parseFromSlice(std.json.Value, allocator, vocab_bytes, .{});
        defer json.deinit();

        if (json.value != .object) {
            _ = try std.io.getStdErr().write("Unable to parse vocab.json.\n");
        }
        const json_root = json.value.object;
        // Reserve so we avoid repeated rehashes
        const kv_count: u32 = @intCast(json_root.count());
        self.*.vocab.ensureTotalCapacity(kv_count) catch {
            _ = try std.io.getStdErr().write("Unable ensure size of self.*.vocab while parsing.\n");
            return error.InvalidJson;
        };

        // Iterate entries: key is []const u8, value is integer
        var iter = json_root.iterator();
        while (iter.next()) |entry| {
            const key = entry.key_ptr.*;

            // Check if the value is an integer
            if (entry.value_ptr.* != .integer) {
                _ = try std.io.getStdErr().writer().print("Found value for key '{s}' is not an integer while parsing vocab. Ignoring value and continue parsing.\n", .{key});
                continue;
            }

            const id: u32 = @intCast(entry.value_ptr.*.integer);

            // Duplicate key into allocator memory
            const dup = try allocator.dupe(u8, key);
            try self.*.vocab.put(dup, id);
        }

        std.debug.print("Finished loadVocab\n", .{});
    }

    fn loadMerges(self: *Tokenizer, allocator: std.mem.Allocator) !void {
        const file = try std.fs.cwd().openFile(self.*.merges_path, .{});
        defer file.close();
        var reader = file.reader();
        var line_buf: [1024]u8 = undefined;
        var prio: usize = 0;
        while (true) {
            const line = try reader.readUntilDelimiter(line_buf[0..], '\n');
            if (line.len == 0) break;
            const trimmed = std.mem.trim(u8, line, " \t\r\n");
            if (trimmed.len == 0 or std.mem.startsWith(u8, trimmed, "#")) continue;
            const idx = std.mem.indexOf(u8, trimmed, " ") orelse continue;
            const a = trimmed[0..idx];
            const b = std.mem.trim(u8, trimmed[idx + 1 ..], " \t\r\n");
            // Insert into the two-level map: merges[a][b] = priority
            if (self.merges.get(a)) |inner| {
                const bdup = try allocator.dupe(u8, b);
                try inner.put(bdup, prio);
            } else {
                var newInner = std.StringHashMap(usize).init(allocator);
                const adup = try allocator.dupe(u8, a);
                const bdup = try allocator.dupe(u8, b);
                try newInner.put(bdup, prio);
                try self.merges.put(adup, newInner);
            }
            prio += 1;
        }
    }

    pub fn tokenize(
        self: *Tokenizer,
        allocator: std.mem.Allocator,
        input: []const u8,
    ) ![]u32 {
        // Arena for intermediate byte-slices.
        var arena = std.heap.ArenaAllocator.init(allocator);
        const aalloc = arena.allocator();

        //const normalized_input = try aalloc.alloc(u8, input.len);

        //for (input, 1..input.len + 1) |char, index| {
        //    if (char == ' ') {
        //        normalized_input[index + 1] = 'Ġ';
        //    } else {
        //        normalized_input[index + 1] = char;
        //    }
        //}

        // Prepend space to input.
        const prefixed_len = input.len + 1;
        const prefixed = try aalloc.alloc(u8, prefixed_len);
        prefixed[0] = 0x20;
        std.mem.copyForwards(u8, prefixed[1..], input);

        // Build initial one-byte parts.
        var parts = std.ArrayList([]u8).init(aalloc);
        for (prefixed) |c| {
            const slice = try aalloc.alloc(u8, 1);
            slice[0] = c;
            try parts.append(slice);
        }
        std.debug.print("parts: {any}\n", .{parts.items});

        // BPE merge loop with O(1) lookup.
        var changed = true;
        while (changed and parts.items.len > 1) {
            changed = false;
            var best_i: usize = 0;
            var best_p: usize = std.math.maxInt(usize);
            for (0..parts.items.len - 1) |i| {
                const x = parts.items[i];
                const y = parts.items[i + 1];
                if (self.merges.get(x)) |inner| {
                    if (inner.get(y)) |p| {
                        if (p < best_p) {
                            best_p = p;
                            best_i = i;
                            changed = true;
                        }
                    }
                }
            }
            if (changed) {
                const first = parts.items[best_i];
                const second = parts.items[best_i + 1];
                const merged = try aalloc.alloc(u8, first.len + second.len);
                std.mem.copyForwards(u8, merged[0..first.len], first);
                std.mem.copyForwards(u8, merged[first.len..], second);
                _ = parts.orderedRemove(best_i + 1);
                parts.items[best_i] = merged;
            }
        }

        // Map to IDs using provided allocator for output.
        var out = std.ArrayList(u32).init(aalloc);
        for (parts.items) |p| {
            if (self.vocab.get(p)) |id| {
                try out.append(id);
            } else {
                try out.append(0);
            }
        }

        const result = try allocator.dupe(u32, out.items);
        arena.deinit();
        return result;
    }
};

test "tokenize aaa b" {
    const given_allocator = std.testing.allocator;

    // 1) Prepare vocab and merges (you may load these from files or define inline for the test)
    // Here we assume you have helper functions that load them synchronously.
    //const given_vocab_path = "../../test_data/byte_level_byte_pair_encoding/vocab.json";

    const given_tokenizer_ptr: *Tokenizer = try Tokenizer.init(given_allocator, "dummy_vocab_path.json", "dummy_merge_path.txt");
    defer given_tokenizer_ptr.*.destroy(given_allocator);

    // Manually populate the vocabulary
    try given_tokenizer_ptr.*.vocab.put("a", 0);
    try given_tokenizer_ptr.*.vocab.put("b", 1);
    try given_tokenizer_ptr.*.vocab.put("aa", 2);
    try given_tokenizer_ptr.*.vocab.put("aaa", 3);

    var inner_0 = std.StringHashMap(usize).init(given_allocator);
    try inner_0.put("a", 0);
    try given_tokenizer_ptr.*.merges.put("a", inner_0); // rule a -> a

    var inner_1 = std.StringHashMap(usize).init(given_allocator);
    try inner_1.put("aa", 1);
    try given_tokenizer_ptr.*.merges.put("aa", inner_1);

    // 2) The input string to test
    const given_input = "aaa b";

    // 3) The expected token ID sequence
    //    You need to run the real tokenizer once (e.g., in Python) to know these IDs.
    //    Here’s an example dummy sequence; replace with the real one:
    const expected_ids = [_]u32{ 3, 1 };

    // 4) Tokenize and compare
    const actual_ids = try given_tokenizer_ptr.*.tokenize(given_allocator, given_input);
    defer given_allocator.free(actual_ids);

    // Assert same length
    std.debug.print("expected:{any} actual:{any}\n", .{ expected_ids, actual_ids });
    try std.testing.expectEqual(actual_ids.len, expected_ids.len);

    // Assert each ID matches
    for (actual_ids, 0..) |id, i| {
        try std.testing.expectEqual(id, expected_ids[i]);
    }
}
