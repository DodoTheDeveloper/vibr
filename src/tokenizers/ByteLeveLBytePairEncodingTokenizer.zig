const std = @import("std");

const MergePair = struct {
    a: []u8,
    b: []u8,
};

pub const Tokenizer = struct {
    vocab_path: []const u8,
    merges_path: []const u8,
    vocab: std.StringHashMap(u32),
    merges: std.ArrayList(MergePair),

    pub fn init(allocator: std.mem.Allocator, vocab_path: []const u8, merges_path: []const u8) !*Tokenizer {
        const tokenizer_ptr = try allocator.create(Tokenizer);
        const _vocab_path = try allocator.alloc(u8, vocab_path.len);
        std.mem.copyForwards(u8, _vocab_path, vocab_path);
        const _merges_path = try allocator.alloc(u8, merges_path.len);
        std.mem.copyForwards(u8, _merges_path, merges_path);
        tokenizer_ptr.* = .{ .vocab_path = _vocab_path, .merges_path = _merges_path, .vocab = std.StringHashMap(u32).init(allocator), .merges = std.ArrayList(MergePair).init(allocator) };

        return tokenizer_ptr;
    }
    pub fn deinit(self: *Tokenizer, allocator: std.mem.Allocator) void {
        allocator.free(self.*.vocab_path);
        allocator.free(self.*.merges_path);

        var vocab_iter = self.*.vocab.iterator();
        while (vocab_iter.next()) |entry| {
            allocator.free(entry.key_ptr.*);
        }
        self.*.vocab.deinit();

        for (self.*.merges.items) |item| {
            allocator.free(item.a);
            allocator.free(item.b);
        }
        self.merges.deinit();
        allocator.destroy(self);
    }

    fn loadVocab(self: *Tokenizer, allocator: std.mem.Allocator) !void {
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
    }

    fn loadMerges(self: *Tokenizer, allocator: std.mem.Allocator) !void {
        const merge_file = try std.fs.cwd().openFile(self.*.merges_path, .{});
        defer merge_file.close();
        const merge_reader = merge_file.reader();
        var line_buffer: [1024]u8 = undefined;

        while (true) {
            const line = try merge_reader.readUntilDelimiter(line_buffer[0..line_buffer.len], '\n');
            if (line.len == 0) break;
            // Trim whitespace and CR/LF
            const trimmed_line = std.mem.trim(u8, line, " \t\r\n");
            if (trimmed_line.len == 0 or std.mem.startsWith(u8, trimmed_line, "#")) {
                continue;
            }

            // Split on the first space: "tokenA tokenB"
            const idx = std.mem.indexOf(u8, trimmed_line, " ") orelse {
                continue;
            };
            const a_bytes_slice = trimmed_line[0..idx];
            const b_bytes_slice_raw = trimmed_line[idx + 1 .. trimmed_line.len];
            const b_bytes_slice = std.mem.trim(u8, b_bytes_slice_raw, " \t\r\n");

            const a_bytes = try allocator.dupe(u8, a_bytes_slice);
            const b_bytes = try allocator.dupe(u8, b_bytes_slice);

            try self.*.merges.append(.{ .a = a_bytes, .b = b_bytes });
        }
    }

    pub fn tokenize(self: *Tokenizer, allocator: std.mem.Allocator, input: []const u8) ![]u32 {
        std.debug.print("{s}", .{input});
        if (self.*.vocab.count() == 0) {
            try self.*.loadVocab(allocator);
        }
        if (self.*.merges.items.len == 0) {
            try self.*.loadMerges(allocator);
        }

        // Implement byte-level BPE tokenization
        var tokens = std.ArrayList(u32).init(allocator);
        defer tokens.deinit();

        // 1. Preprocess input - add spaces before non-space characters
        var preprocessed = std.ArrayList(u8).init(allocator);
        defer preprocessed.deinit();

        var i: usize = 0;
        while (i < input.len) {
            // Add space before token if not at beginning and previous char isn't whitespace
            if (i > 0 and !std.ascii.isWhitespace(input[i - 1]) and !std.ascii.isWhitespace(input[i])) {
                try preprocessed.append(' ');
            }
            try preprocessed.append(input[i]);
            i += 1;
        }

        // 2. Tokenize the preprocessed input
        var words = std.ArrayList([]u8).init(allocator);
        defer {
            for (words.items) |word| {
                allocator.free(word);
            }
            words.deinit();
        }

        // Split input into words (by whitespace)
        var iter = std.mem.splitAny(u8, preprocessed.items, " ");
        while (iter.next()) |word| {
            if (word.len == 0) continue;

            // Create a copy of the word
            const word_copy = try allocator.dupe(u8, word);
            try words.append(word_copy);
        }

        // 3. Apply BPE merges to each word
        for (words.items) |word| {
            // Start with characters as separate tokens
            var parts = std.ArrayList([]u8).init(allocator);
            defer {
                for (parts.items) |part| {
                    allocator.free(part);
                }
                parts.deinit();
            }

            // Initialize with individual bytes
            for (word) |byte| {
                var byte_token = try allocator.alloc(u8, 1);
                byte_token[0] = byte;
                try parts.append(byte_token);
            }

            // Apply merges according to priority
            var merged = true;
            while (merged and parts.items.len > 1) {
                merged = false;

                // Find best merge
                var best_merge_idx: ?usize = null;
                var best_merge_priority: usize = std.math.maxInt(usize);

                for (0..parts.items.len - 1) |idx| {
                    // Check if this pair can be merged
                    const pair_a = parts.items[idx];
                    const pair_b = parts.items[idx + 1];

                    // Find this pair in merges list
                    for (self.*.merges.items, 0..) |merge, merge_idx| {
                        if (std.mem.eql(u8, merge.a, pair_a) and std.mem.eql(u8, merge.b, pair_b)) {
                            // Found a merge, check priority
                            if (merge_idx < best_merge_priority) {
                                best_merge_priority = merge_idx;
                                best_merge_idx = idx;
                            }
                        }
                    }
                }

                // Apply best merge if found
                if (best_merge_idx) |idx| {
                    const pair_a = parts.items[idx];
                    const pair_b = parts.items[idx + 1];

                    // Create merged token
                    const merged_token = try allocator.alloc(u8, pair_a.len + pair_b.len);
                    std.mem.copyForwards(u8, merged_token, pair_a);
                    std.mem.copyForwards(u8, merged_token[pair_a.len..], pair_b);

                    // Replace the two tokens with the merged one
                    allocator.free(pair_a);
                    allocator.free(pair_b);
                    parts.items[idx] = merged_token;
                    _ = parts.orderedRemove(idx + 1);

                    merged = true;
                }
            }

            // Convert final tokens to IDs
            for (parts.items) |part| {
                if (self.*.vocab.get(part)) |id| {
                    try tokens.append(id);
                } else {
                    // Handle unknown token - use a special token ID or byte fallback
                    // For now, just use 0 as unknown token ID
                    try tokens.append(0);
                }
            }
        }

        // Return the token IDs
        return allocator.dupe(u32, tokens.items);
    }
};

test "tokenize Hello, Qwen3 tokenizer!" {
    const given_allocator = std.testing.allocator;

    // 1) Prepare vocab and merges (you may load these from files or define inline for the test)
    // Here we assume you have helper functions that load them synchronously.
    //const given_vocab_path = "../../test_data/byte_level_byte_pair_encoding/vocab.json";

    const given_tokenizer_ptr: *Tokenizer = try Tokenizer.init(given_allocator, "dummy_vocab_path.json", "dummy_merge_path.txt");
    defer given_tokenizer_ptr.*.deinit(given_allocator);

    // Manually populate the vocabulary
    try given_tokenizer_ptr.*.vocab.put("Hello", 15496);
    try given_tokenizer_ptr.*.vocab.put(",", 11);
    try given_tokenizer_ptr.*.vocab.put("Ġ", 50256);
    try given_tokenizer_ptr.*.vocab.put("Q", 81);
    try given_tokenizer_ptr.*.vocab.put("wen", 119);
    try given_tokenizer_ptr.*.vocab.put("3", 52);
    try given_tokenizer_ptr.*.vocab.put("token", 2040);
    try given_tokenizer_ptr.*.vocab.put("izer", 1054);
    try given_tokenizer_ptr.*.vocab.put("!", 0);

    const given_merge1 = MergePair{
        .a = try given_allocator.dupe(u8, "t"),
        .b = try given_allocator.dupe(u8, "o"),
    };
    try given_tokenizer_ptr.*.merges.append(given_merge1);

    const given_merge2 = MergePair{
        .a = try given_allocator.dupe(u8, "to"),
        .b = try given_allocator.dupe(u8, "k"),
    };
    try given_tokenizer_ptr.*.merges.append(given_merge2);

    const given_merge3 = MergePair{
        .a = try given_allocator.dupe(u8, "tok"),
        .b = try given_allocator.dupe(u8, "e"),
    };
    try given_tokenizer_ptr.*.merges.append(given_merge3);

    const given_merge4 = MergePair{
        .a = try given_allocator.dupe(u8, "toke"),
        .b = try given_allocator.dupe(u8, "n"),
    };
    try given_tokenizer_ptr.*.merges.append(given_merge4);

    // 2) The input string to test
    const given_input = "Hello, Qwen3 tokenizer!";

    // 3) The expected token ID sequence
    //    You need to run the real tokenizer once (e.g., in Python) to know these IDs.
    //    Here’s an example dummy sequence; replace with the real one:
    const expected_ids = [_]u32{
        15496, // "Hello"
        11, // ","
        50256, // "Ġ"
        81, // "Q"
        119, // "wen"
        52, // "3"
        50256, // "Ġ"
        2040, // "token"
        1054, // "izer"
        0, // "!" (end-of-text or punctuation token)
    };

    // 4) Tokenize and compare
    const actual_ids = try given_tokenizer_ptr.*.tokenize(given_allocator, given_input);
    defer given_allocator.free(actual_ids);

    // Assert same length
    try std.testing.expectEqual(actual_ids.len, expected_ids.len);

    // Assert each ID matches
    for (actual_ids, 0..) |id, i| {
        try std.testing.expectEqual(id, expected_ids[i]);
    }
}
