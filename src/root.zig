const std = @import("std");
const parsers_term = @import("parsers/term.zig");
const utils = @import("utils.zig");
const prompts_formatter = @import("prompts/formatter.zig");
const requests = @import("requests.zig");
const testing = std.testing;

// Import & add modules where tests should automatically run with the command:
//     `zig build test --summary new`
// Why? -> https://ziggit.dev/t/how-do-i-get-zig-build-to-run-all-the-tests/4434/3
test "run all tests" {
    _ = parsers_term;
    _ = utils;
    _ = prompts_formatter;
    _ = requests;
}
