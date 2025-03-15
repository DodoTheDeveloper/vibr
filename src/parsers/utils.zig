const std = @import("std");
pub fn countLeadingWhitespace(s: []const u8) usize {
    var count: usize = 0;
    while (count < s.len and (s[count] == ' ' or s[count] == '\t')) : (count += 1) {}
    return count;
}

test "counts leading whitespaces" {
    const line = "  \ttwowhitespacesonetab";
    const num_whitespaces = countLeadingWhitespace(line);
    try std.testing.expectEqual(3, num_whitespaces);
}
