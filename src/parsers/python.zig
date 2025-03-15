const std = @import("std");
const parser_utils = @import("utils.zig");

const PythonParserError = error{ConstructNotImplemented};

const PythonNodeType = union(enum) {
    file,
    function,
    class,
};

const FileData = struct {
    file_lines: std.ArrayList([]const u8),
    pub fn deinit(self: *FileData) void {
        self.*.file_lines.deinit();
    }
};

const PythonNode = struct {
    parent: ?*PythonNode = null,
    children: std.ArrayList(*PythonNode),
    node_type: PythonNodeType,
    file_data: *FileData,
    indent: usize = 0,
    line_start_idx: usize = 0,
    line_end_idx: usize = 0,

    pub fn new(allocator: std.mem.Allocator, parent: ?*PythonNode, node_type: PythonNodeType, file_data: *FileData, indent: usize, line_start: usize) !*PythonNode {
        const python_node = try allocator.create(PythonNode);
        python_node.* = PythonNode{ .parent = parent, .children = std.ArrayList(*PythonNode).init(allocator), .node_type = node_type, .file_data = file_data, .indent = indent, .line_start_idx = line_start };
        return python_node;
    }

    pub fn deinit(self: *PythonNode, allocator: std.mem.Allocator) PythonParserError!void {
        for (self.*.children.items) |child| {
            try child.*.deinit(allocator);
            allocator.destroy(child);
        }
        self.*.children.deinit();

        if (self.parent == null) { // is root
            self.*.file_data.deinit();
            allocator.destroy(self.file_data);
        }
    }

    pub fn getNodeContent(self: *PythonNode, allocator: std.mem.Allocator) ![]const u8 {
        const myLines = self.file_data.file_lines.items[self.line_start_idx .. self.line_end_idx + 1];
        var result = std.ArrayList(u8).init(allocator);
        for (myLines, 0..) |line, idx| {
            try result.appendSlice(line[self.indent..line.len]);
            if (idx != myLines.len - 1) { // only add newline if not last item
                try result.append('\n');
            }
        }

        return result.toOwnedSlice();
    }
};

fn parseCodeLines(allocator: std.mem.Allocator, code: []const u8) !std.ArrayList([]const u8) {
    var line_iter = std.mem.splitAny(u8, code, "\n");
    var lines = std.ArrayList([]const u8).init(allocator);
    while (line_iter.next()) |line| {
        try lines.append(line);
    }
    return lines;
}

fn startsWithPythonKeyword(line: []const u8) bool {
    return std.mem.startsWith(u8, line, "class") or std.mem.startsWith(u8, line, "def");
}

fn getNodeTypeFromKeyword(line: []const u8) !PythonNodeType {
    var node_type: PythonNodeType = .class;
    if (std.mem.startsWith(u8, line, "class")) {
        node_type = .class;
    } else if (std.mem.startsWith(u8, line, "def")) {
        node_type = .function;
    } else {
        return PythonParserError.ConstructNotImplemented;
    }
    return node_type;
}

pub fn parseFile(allocator: std.mem.Allocator, code: []const u8) !*PythonNode {
    const file_data = try allocator.create(FileData);
    file_data.* = FileData{ .file_lines = std.ArrayList([]const u8).init(allocator) };

    var line_iter = std.mem.splitAny(u8, code, "\n");
    var current_indent: usize = 0;
    var node_type: PythonNodeType = .file;
    const root_node_ptr = try PythonNode.new(
        allocator,
        null,
        node_type,
        file_data,
        current_indent,
        0,
    );

    var current_node_ptr: *PythonNode = root_node_ptr;

    var line_count: usize = 0;
    while (true) {
        const maybe_line = line_iter.next();
        if (maybe_line == null) {
            break;
        }
        const raw_line = maybe_line.?;
        try root_node_ptr.file_data.file_lines.append(raw_line);

        current_indent = parser_utils.countLeadingWhitespace(raw_line);

        // block closed, set parent to current_node, could be multiple
        while (current_indent <= current_node_ptr.*.indent and current_node_ptr.*.parent != null) {
            current_node_ptr.*.line_end_idx = if (line_count > 0) line_count - 1 else 0;
            current_node_ptr = current_node_ptr.parent.?;
        }

        const trimmed_line = std.mem.trim(u8, raw_line, " \t"); // whitespace & tabs
        if (startsWithPythonKeyword(trimmed_line)) {
            node_type = try getNodeTypeFromKeyword(trimmed_line);

            const new_node_ptr = try PythonNode.new(allocator, current_node_ptr, node_type, file_data, current_indent, line_count);
            try current_node_ptr.*.children.append(new_node_ptr);

            current_node_ptr = new_node_ptr;
        }

        line_count += 1;
    }
    // std.mem.split will return one more element than there are lines due to the split on the
    // last \n.
    root_node_ptr.*.line_end_idx = line_count - 2;

    return root_node_ptr;
}

test "parses a python file" {
    const given_allocator = std.testing.allocator;

    const file_path = "./test_data/python/parser_test.py";
    const file = try std.fs.cwd().openFile(file_path, .{});
    const given_python_code = try file.readToEndAlloc(given_allocator, 2048);
    defer given_allocator.free(given_python_code);

    // test root node
    const actual_root_node_ptr = try parseFile(given_allocator, given_python_code);
    defer given_allocator.destroy(actual_root_node_ptr);

    const expected_root_content =
        \\class SomeClass():
        \\    def inner_function(self):
        \\        print("some")
        \\
        \\
        \\def some_function():
        \\    class InnerClass():
        \\        pass
    ;
    const actual_root_node_content = try actual_root_node_ptr.*.getNodeContent(given_allocator);
    defer given_allocator.free(actual_root_node_content);
    try std.testing.expectEqualSlices(u8, expected_root_content, actual_root_node_content);

    try std.testing.expect(actual_root_node_ptr.*.parent == null and actual_root_node_ptr.*.children.items.len == 2 and actual_root_node_ptr.*.indent == 0 and actual_root_node_ptr.*.node_type == .file and actual_root_node_ptr.*.line_start_idx == 0 and actual_root_node_ptr.*.line_end_idx == 7);

    // test child 0
    const actual_child_0_ptr = actual_root_node_ptr.children.items[0];

    const expected_child_0_content =
        \\class SomeClass():
        \\    def inner_function(self):
        \\        print("some")
    ;

    const actual_expected_child_0_content = try actual_child_0_ptr.*.getNodeContent(given_allocator);
    defer given_allocator.free(actual_expected_child_0_content);
    try std.testing.expectEqualSlices(u8, expected_child_0_content, actual_expected_child_0_content);
    try std.testing.expect(actual_child_0_ptr.*.parent == actual_root_node_ptr and actual_child_0_ptr.children.items.len == 1 and actual_child_0_ptr.*.indent == 0 and actual_child_0_ptr.*.node_type == .class and actual_child_0_ptr.*.line_start_idx == 0 and actual_child_0_ptr.*.line_end_idx == 2);

    // test child 0 of child 0
    const actual_child_0_0_ptr = actual_child_0_ptr.children.items[0];

    const expected_child_0_0_content =
        \\def inner_function(self):
        \\    print("some")
    ;
    const actual_child_0_0_content = try actual_child_0_0_ptr.*.getNodeContent(given_allocator);
    defer given_allocator.free(actual_child_0_0_content);
    try std.testing.expectEqualSlices(u8, expected_child_0_0_content, actual_child_0_0_content);
    try std.testing.expect(actual_child_0_0_ptr.*.parent == actual_child_0_ptr and actual_child_0_0_ptr.children.items.len == 0 and actual_child_0_0_ptr.*.indent == 4 and actual_child_0_0_ptr.*.node_type == .function and actual_child_0_0_ptr.*.line_start_idx == 1 and actual_child_0_0_ptr.*.line_end_idx == 2);

    const expected_child_1_content =
        \\def some_function():
        \\    class InnerClass():
        \\        pass
    ;
    const actual_child_1_ptr = actual_root_node_ptr.*.children.items[1];
    const actual_child_1_content = try actual_child_1_ptr.*.getNodeContent(given_allocator);
    defer given_allocator.free(actual_child_1_content);
    try std.testing.expectEqualSlices(u8, expected_child_1_content, actual_child_1_content);

    try std.testing.expect(actual_child_1_ptr.*.parent == actual_root_node_ptr and actual_child_1_ptr.children.items.len == 1 and actual_child_1_ptr.*.indent == 0 and actual_child_1_ptr.*.node_type == .function and actual_child_1_ptr.*.line_start_idx == 5 and actual_child_1_ptr.*.line_end_idx == 7);

    const expected_child_1_0_content =
        \\class InnerClass():
        \\    pass
    ;
    const actual_child_1_0_ptr = actual_child_1_ptr.children.items[0];
    const actual_child_1_0_content = try actual_child_1_0_ptr.*.getNodeContent(given_allocator);
    defer given_allocator.free(actual_child_1_0_content);
    try std.testing.expectEqualSlices(u8, expected_child_1_0_content, actual_child_1_0_content);

    try std.testing.expect(actual_child_1_0_ptr.*.parent == actual_child_1_ptr and actual_child_1_0_ptr.children.items.len == 0 and actual_child_1_0_ptr.*.indent == 4 and actual_child_1_0_ptr.*.node_type == .class and actual_child_1_0_ptr.*.line_start_idx == 6 and actual_child_1_0_ptr.*.line_end_idx == 7);

    try actual_root_node_ptr.deinit(given_allocator);
}
