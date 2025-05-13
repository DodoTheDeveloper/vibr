const std = @import("std");

/// Build the GPT-2 “byte-to-Unicode” table at comptime.
/// Each element is the UTF-8 representation of the stand-in code point.
pub fn create_bytes_to_unicode_map_gpt2(allocator: std.mem.Allocator) !*[256][]const u8 {
    // Returns a slice which maps possible values of a byte (0-255) in such a way that control characters
    // between 0-32, 127-160 and 173 are escaped. The first control character gets escaped with the
    // unicode character U+0100, the next one with U+0101 and so on. This is a form of lossless compression
    // used to map all 255 values of a byte which then can for example be used in a json body.
    var bytes: [256]u16 = undefined;
    const map_ptr: *[256][]const u8 = try allocator.create([256][]const u8);

    var extra_count: u16 = 0;
    const UNICODE_START: u16 = 256;
    var buf: [4]u8 = undefined;

    // 0-32 control chars -> encode
    inline for (0..33) |b| {
        bytes[b] = UNICODE_START + extra_count;

        // add to output map
        const byte_written = std.unicode.utf8Encode(bytes[b], &buf) catch unreachable;
        map_ptr.*[b] = try allocator.dupe(u8, buf[0..byte_written]); // index by original byte value
        extra_count += 1;
    }

    // 33-126 safe chars
    inline for (33..127) |b| {
        bytes[b] = @intCast(b);
        // add to output map
        const byte_written = std.unicode.utf8Encode(bytes[b], &buf) catch unreachable;
        map_ptr.*[b] = try allocator.dupe(u8, buf[0..byte_written]); // index by original byte value
    }

    // 127 - 160 control chars
    inline for (127..161) |b| {
        bytes[b] = UNICODE_START + extra_count;

        // add to output map
        const byte_written = std.unicode.utf8Encode(bytes[b], &buf) catch unreachable;
        map_ptr.*[b] = try allocator.dupe(u8, buf[0..byte_written]); // index by original byte value
        extra_count += 1;
    }

    // 161 - 172 safe chars
    inline for (161..173) |b| {
        bytes[b] = @intCast(b);
        // add to output map
        const byte_written = std.unicode.utf8Encode(bytes[b], &buf) catch unreachable;
        map_ptr.*[b] = try allocator.dupe(u8, buf[0..byte_written]); // index by original byte value
    }

    // 173 control char
    inline for (173..174) |b| {
        const unicode = UNICODE_START + extra_count;
        bytes[b] = @intCast(unicode);
        // add to output map
        const byte_written = std.unicode.utf8Encode(bytes[b], &buf) catch unreachable;
        map_ptr.*[b] = try allocator.dupe(u8, buf[0..byte_written]); // index by original byte value
        extra_count += 1;
    }

    // 174 - 255 safe chars
    inline for (174..256) |b| {
        bytes[b] = @intCast(b);
        const byte_written = std.unicode.utf8Encode(bytes[b], &buf) catch unreachable;
        map_ptr.*[b] = try allocator.dupe(u8, buf[0..byte_written]); // index by original byte value
    }

    return map_ptr;
}

test "builds byte encoder table" {
    const given_allocator = std.testing.allocator;
    const expected: [256][]const u8 = .{
        "Ā", "ā", "Ă", "ă", "Ą", "ą", "Ć", "ć", "Ĉ", "ĉ", "Ċ", "ċ", "Č", "č", "Ď", "ď",
        "Đ", "đ", "Ē", "ē", "Ĕ", "ĕ", "Ė", "ė", "Ę", "ę", "Ě", "ě", "Ĝ", "ĝ", "Ğ", "ğ",
        "Ġ", "!",  "\"", "#",  "$",  "%",  "&",  "'",  "(",  ")",  "*",  "+",  ",",  "-",  ".",  "/",
        "0",  "1",  "2",  "3",  "4",  "5",  "6",  "7",  "8",  "9",  ":",  ";",  "<",  "=",  ">",  "?",
        "@",  "A",  "B",  "C",  "D",  "E",  "F",  "G",  "H",  "I",  "J",  "K",  "L",  "M",  "N",  "O",
        "P",  "Q",  "R",  "S",  "T",  "U",  "V",  "W",  "X",  "Y",  "Z",  "[",  "\\", "]",  "^",  "_",
        "`",  "a",  "b",  "c",  "d",  "e",  "f",  "g",  "h",  "i",  "j",  "k",  "l",  "m",  "n",  "o",
        "p",  "q",  "r",  "s",  "t",  "u",  "v",  "w",  "x",  "y",  "z",  "{",  "|",  "}",  "~",  "ġ",
        "Ģ", "ģ", "Ĥ", "ĥ", "Ħ", "ħ", "Ĩ", "ĩ", "Ī", "ī", "Ĭ", "ĭ", "Į", "į", "İ", "ı",
        "Ĳ", "ĳ", "Ĵ", "ĵ", "Ķ", "ķ", "ĸ", "Ĺ", "ĺ", "Ļ", "ļ", "Ľ", "ľ", "Ŀ", "ŀ", "Ł",
        "ł", "¡", "¢", "£", "¤", "¥", "¦", "§", "¨", "©", "ª", "«", "¬", "Ń", "®", "¯",
        "°", "±", "²", "³", "´", "µ", "¶", "·", "¸", "¹", "º", "»", "¼", "½", "¾", "¿",
        "À", "Á", "Â", "Ã", "Ä", "Å", "Æ", "Ç", "È", "É", "Ê", "Ë", "Ì", "Í", "Î", "Ï",
        "Ð", "Ñ", "Ò", "Ó", "Ô", "Õ", "Ö", "×", "Ø", "Ù", "Ú", "Û", "Ü", "Ý", "Þ", "ß",
        "à", "á", "â", "ã", "ä", "å", "æ", "ç", "è", "é", "ê", "ë", "ì", "í", "î", "ï",
        "ð", "ñ", "ò", "ó", "ô", "õ", "ö", "÷", "ø", "ù", "ú", "û", "ü", "ý", "þ", "ÿ",
    };

    const encoder_ptr = try create_bytes_to_unicode_map_gpt2(given_allocator);
    defer {
        for (encoder_ptr.*) |slice| {
            given_allocator.free(slice);
        }
        given_allocator.free(encoder_ptr);
    }

    for (encoder_ptr.*, 0..) |actual, index| {
        std.testing.expectEqualSlices(u8, expected[index], actual) catch continue;
    }
}
