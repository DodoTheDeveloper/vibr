const std = @import("std");
const encoder = @import("./../utils/byte_to_unicode_gpt2.zig").create_bytes_to_unicode_map_gpt2;

pub const Tokenizer = struct {
    vocab_path: []const u8,
    merges_path: []const u8,
    vocab: std.StringHashMap(u32),
    inv_vocab: std.AutoHashMap(u32, []const u8), // shared memory with vocab
    merges: std.StringHashMap(std.StringHashMap(usize)),

    pub fn init(allocator: std.mem.Allocator, vocab_path: []const u8, merges_path: []const u8) !*Tokenizer {
        const tokenizer_ptr = try allocator.create(Tokenizer);
        const _vocab_path = try allocator.alloc(u8, vocab_path.len);
        std.mem.copyForwards(u8, _vocab_path, vocab_path);
        const _merges_path = try allocator.alloc(u8, merges_path.len);
        std.mem.copyForwards(u8, _merges_path, merges_path);
        tokenizer_ptr.* = .{ .vocab_path = _vocab_path, .merges_path = _merges_path, .vocab = std.StringHashMap(u32).init(allocator), .merges = std.StringHashMap(std.StringHashMap(usize)).init(allocator), .inv_vocab = std.AutoHashMap(u32, []const u8).init(allocator) };

        return tokenizer_ptr;
    }
    pub fn destroy(self: *Tokenizer, allocator: std.mem.Allocator) void {
        std.debug.print("DESTROY 0\n", .{});
        allocator.free(self.*.vocab_path);
        allocator.free(self.*.merges_path);

        std.debug.print("DESTROY 1\n", .{});
        var vocab_iter = self.*.vocab.iterator();
        while (vocab_iter.next()) |entry| {
            allocator.free(entry.key_ptr.*);
        }
        std.debug.print("DESTROY 2\n", .{});
        self.*.vocab.deinit();
        self.*.inv_vocab.deinit();
        std.debug.print("DESTROY 3\n", .{});

        var it = self.merges.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.*.deinit();
        }

        std.debug.print("DESTROY 4\n", .{});
        self.*.merges.deinit();
        allocator.destroy(self);
        std.debug.print("DESTROY 5\n", .{});
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
            try self.*.inv_vocab.put(id, dup);
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
            std.debug.print("MERGES 0 \n", .{});
            const maybe_line = try reader.readUntilDelimiterOrEof(line_buf[0..], '\n');
            std.debug.print("MERGES 1 {any}\n", .{maybe_line});

            if (maybe_line == null) {
                std.debug.print("MERGES 40 \n", .{});
                break;
            }
            const line = maybe_line.?;
            std.debug.print("MERGES 2 {s}\n", .{line});
            const trimmed = std.mem.trim(u8, line, " \t\r\n");
            if (trimmed.len == 0 or std.mem.startsWith(u8, trimmed, "#")) continue;
            const idx = std.mem.indexOf(u8, trimmed, " ") orelse continue;
            std.debug.print("MERGES 10 \n", .{});
            const a = trimmed[0..idx];
            const b = std.mem.trim(u8, trimmed[idx + 1 ..], " \t\r\n");
            std.debug.print("MERGES 20 \n", .{});
            // Insert into the two-level map: merges[a][b] = priority
            if (self.*.merges.getPtr(a)) |inner| {
                std.debug.print("MERGES 21 \n", .{});
                const bdup = try allocator.dupe(u8, b);
                try inner.*.put(bdup, prio);
                std.debug.print("MERGES 24 \n", .{});
            } else {
                std.debug.print("MERGES 25 \n", .{});
                var newInner = std.StringHashMap(usize).init(allocator);
                const adup = try allocator.dupe(u8, a);
                const bdup = try allocator.dupe(u8, b);
                try newInner.put(bdup, prio);
                try self.*.merges.put(adup, newInner);
                std.debug.print("MERGES 27 \n", .{});
            }
            prio += 1;
            std.debug.print("MERGES 29 \n", .{});
        }
        std.debug.print("MERGES 30 \n", .{});
    }

    pub fn tokenize(
        self: *Tokenizer,
        allocator: std.mem.Allocator,
        input: []const u8,
    ) ![]u32 {
        // Arena for intermediate byte-slices.
        var arena = std.heap.ArenaAllocator.init(allocator);
        const aalloc = arena.allocator();

        if (self.*.vocab.count() == 0) {
            self.*.loadVocab(allocator) catch {
                _ = try std.io.getStdErr().write("Unable to load vocab.json file ");
                _ = try std.io.getStdErr().write(self.*.vocab_path);
                _ = try std.io.getStdErr().write(".\n");
            };
        }
        if (self.*.merges.count() == 0) {
            self.*.loadMerges(allocator) catch {
                _ = try std.io.getStdErr().write("Unable to load merges.txt file ");
                _ = try std.io.getStdErr().write(self.*.merges_path);
                _ = try std.io.getStdErr().write(".\n");
            };
        }

        // normalize input data
        var normalized_input_list = std.ArrayList(u8).init(allocator);
        defer normalized_input_list.deinit();
        const normalizer_ptr = try encoder(allocator);
        defer {
            for (normalizer_ptr.*) |entry| {
                allocator.free(entry);
            }
            allocator.destroy(normalizer_ptr);
        }

        for (input) |char| {
            const codepoint = normalizer_ptr.*[char];
            for (codepoint) |byte| {
                try normalized_input_list.append(byte);
            }
        }

        const normalized_input = normalized_input_list.items;
        // Prepend space to input.

        std.debug.print("[DODO] Normalized input {s}\n", .{normalized_input});
        // Build initial one-byte parts.
        var parts = std.ArrayList([]u8).init(aalloc);
        for (normalized_input) |c| {
            const slice = try aalloc.alloc(u8, 1);
            slice[0] = c;
            try parts.append(slice);
        }
        std.debug.print("parts: {s}\n", .{parts.items});

        // BPE merge loop with O(1) lookup.
        var changed = true;
        while (changed and parts.items.len > 1) {
            changed = false;
            var best_i: usize = 0;
            var best_p: usize = std.math.maxInt(usize);
            for (0..parts.items.len - 1) |i| {
                const x = parts.items[i];
                const y = parts.items[i + 1];
                std.debug.print("loop x {s} y {s}\n", .{ x, y });
                if (self.*.merges.get(x)) |inner| {
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

    pub fn detokenize(self: *Tokenizer, allocator: std.mem.Allocator, input: []const u32) ![]const u8 {
        std.debug.print("{any} {any} {any}", .{ self, allocator, input });

        // load vocab & merges if not loaded yet.
        if (self.*.vocab.count() == 0) {
            self.*.loadVocab(allocator) catch {
                _ = try std.io.getStdErr().write("Unable to load vocab.json file ");
                _ = try std.io.getStdErr().write(self.*.vocab_path);
                _ = try std.io.getStdErr().write(".\n");
            };
        }
        if (self.*.merges.count() == 0) {
            self.*.loadMerges(allocator) catch {
                _ = try std.io.getStdErr().write("Unable to load merges.txt file ");
                _ = try std.io.getStdErr().write(self.*.merges_path);
                _ = try std.io.getStdErr().write(".\n");
            };
        }

        var inv_vocab = std.AutoHashMap(u32, []const u8).init(allocator);

        var vocab_iter = self.*.vocab.iterator();

        while (vocab_iter.next()) |entry| {
            try inv_vocab.put(entry.value_ptr.*, entry.key_ptr.*);
        }

        const some = "hello";
        const out = try allocator.dupe(u8, some);
        return out;
    }
};

test "tokenize aaab" {
    std.debug.print("DODO 1\n", .{});
    const given_allocator = std.testing.allocator;

    // 1) Prepare vocab and merges (you may load these from files or define inline for the test)
    // Here we assume you have helper functions that load them synchronously.
    //const given_vocab_path = "../../test_data/byte_level_byte_pair_encoding/vocab.json";

    const given_tokenizer_ptr: *Tokenizer = try Tokenizer.init(given_allocator, "./test_data/mini_vocab.json", "./test_data/mini_merges.txt");
    defer given_tokenizer_ptr.*.destroy(given_allocator);

    std.debug.print("DODO 2\n", .{});
    // 2) The input string to test
    const given_input = "aaab";

    // 3) The expected token ID sequence
    //    You need to run the real tokenizer once (e.g., in Python) to know these IDs.
    //    Here’s an example dummy sequence; replace with the real one:
    const expected_ids = [_]u32{ 3, 1 };

    std.debug.print("DODO 3\n", .{});
    // 4) Tokenize and compare
    const actual_ids = try given_tokenizer_ptr.*.tokenize(given_allocator, given_input);
    defer given_allocator.free(actual_ids);
    std.debug.print("DODO 4 {any}\n", .{actual_ids});

    // Assert same length
    try std.testing.expectEqual(actual_ids.len, expected_ids.len);

    std.debug.print("DODO 5\n", .{});
    // Assert each ID matches
    for (actual_ids, 0..) |id, i| {
        try std.testing.expectEqual(id, expected_ids[i]);
    }

    std.debug.print("DODO 6\n", .{});
}

test "detokenize" {
    std.debug.print("Unitl here 0\n", .{});
    const given_allocator = std.testing.allocator;

    // 1) Prepare vocab and merges (you may load these from files or define inline for the test)
    // Here we assume you have helper functions that load them synchronously.
    //const given_vocab_path = "../../test_data/byte_level_byte_pair_encoding/vocab.json";

    const given_tokenizer_ptr: *Tokenizer = try Tokenizer.init(given_allocator, "./test_data/mini_vocab.json", "./test_data/mini_merges.txt");
    defer given_tokenizer_ptr.*.destroy(given_allocator);

    // 3) The expected token ID sequence
    //    You need to run the real tokenizer once (e.g., in Python) to know these IDs.
    //    Here’s an example dummy sequence; replace with the real one:
    const given_tokens = [_]u32{ 3, 1 };
    const expected_output = "aaab";

    // 4) Tokenize and compare
    const actual_output = try given_tokenizer_ptr.*.detokenize(given_allocator, &given_tokens);
    defer given_allocator.free(actual_output);

    // Assert same length
    try std.testing.expectEqual(expected_output, actual_output);
}
