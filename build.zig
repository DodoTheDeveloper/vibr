const std = @import("std");

<<<<<<< Updated upstream
// Although this function looks imperative, note that its job is to
// declaratively construct a build graph that will be executed by an external
// runner.
pub fn build(b: *std.Build) void {
    // Standard target options allows the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});

    // Standard optimization options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall. Here we do not
    // set a preferred release mode, allowing the user to decide how to optimize.
    const optimize = b.standardOptimizeOption(.{});

    const target_os = target.result.os.tag;
    const host_os = b.graph.host.result.os.tag;
    const arch = target.result.cpu.arch;
    const ONNX_VERSION = "1.21.0";
=======
const ONNX_VERSION = "1.20.1";
fn build_mac_os(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) void {
>>>>>>> Stashed changes
    const ONNX_DIR = "onnxruntime/onnxruntime-osx-universal2-" ++ ONNX_VERSION;
    const ONNX_URL = "https://sourceforge.net/projects/onnx-runtime.mirror/files/v" ++ ONNX_VERSION ++ "/onnxruntime-osx-universal2-" ++ ONNX_VERSION ++ ".tgz/download";

    const exe = b.addExecutable(.{
        .name = "cabal",
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    var onnx_header_path_lazy = b.path("");
    var onnx_lib_path_lazy = b.path("");

    // download onnxruntime if not present
    const fetch_step = b.addSystemCommand(&[_][]const u8{
        "bash",
        "-c",
        // single-line shell script:
        "if [ ! -d \"" ++ ONNX_DIR ++ "\" ]; then " ++
            "mkdir -p onnxruntime && " ++
            "curl -L " ++ ONNX_URL ++ " | tar xz -C onnxruntime; " ++
            "fi",
    });
    // Name the step so it shows up in `zig build --help`.
    fetch_step.step.name = "fetch-onnx";
    exe.step.dependOn(&fetch_step.step);

<<<<<<< Updated upstream
        // download onnxruntime if not present
        const fetch_step = b.addSystemCommand(&[_][]const u8{
            "bash",
            "-c",
            // single-line shell script:
            "if [ ! -d \"" ++ ONNX_DIR ++ "\" ]; then " ++
                "mkdir -p onnxruntime && " ++
                "curl -L " ++ ONNX_URL ++ " | tar xz -C onnxruntime; " ++
                "fi",
        });
        // Name the step so it shows up in `zig build --help`.
        fetch_step.step.name = "fetch-onnx";
        exe.step.dependOn(&fetch_step.step);

        onnx_header_path_lazy = b.path("onnxruntime/onnxruntime-osx-universal2-1.21.0/include");
        const onnx_lib_path = "onnxruntime/onnxruntime-osx-universal2-1.21.0/lib";
        onnx_lib_path_lazy = b.path(onnx_lib_path);
        exe.addRPath(b.path("@loader_path/../" ++ onnx_lib_path));
    }
=======
    onnx_header_path_lazy = b.path("onnxruntime/onnxruntime-osx-universal2-" ++ ONNX_VERSION ++ "/include");
    const onnx_lib_path = "onnxruntime/onnxruntime-osx-universal2-" ++ ONNX_VERSION ++ "/lib";
    onnx_lib_path_lazy = b.path(onnx_lib_path);
    exe.addRPath(b.path("@loader_path/../" ++ onnx_lib_path));
>>>>>>> Stashed changes
    exe.addIncludePath(onnx_header_path_lazy);
    exe.addLibraryPath(onnx_lib_path_lazy);
    exe.linkSystemLibrary("onnxruntime");

    // This declares intent for the executable to be installed into the
    // standard location when the user invokes the "install" step (the default
    // step when running `zig build`).
    b.installArtifact(exe);

    // This *creates* a Run step in the build graph, to be executed when another
    // step is evaluated that depends on it. The next line below will establish
    // such a dependency.
    const run_cmd = b.addRunArtifact(exe);

    // By making the run step depend on the install step, it will be run from the
    // installation directory rather than directly from within the cache directory.
    // This is not necessary, however, if the application depends on other installed
    // files, this ensures they will be present and in the expected location.
    run_cmd.step.dependOn(b.getInstallStep());

    // This allows the user to pass arguments to the application in the build
    // command itself, like this: `zig build run -- arg1 arg2 etc`
    if (b.args) |args| {
        run_cmd.addArgs(args);
    }

    // This creates a build step. It will be visible in the `zig build --help` menu,
    // and can be selected like this: `zig build run`
    // This will evaluate the `run` step rather than the default, which is "install".
    const run_step = b.step("run", "Run the app");
    run_step.dependOn(&run_cmd.step);

    // Creates a step for unit testing. This only builds the test executable
    // but does not run it.
    const lib_unit_tests = b.addTest(.{
        .root_source_file = b.path("src/root.zig"),
        .target = target,
        .optimize = optimize,
    });

    exe.addRPath(b.path("@loader_path/../" ++ onnx_lib_path));
    lib_unit_tests.addRPath(b.path("@loader_path/../" ++ onnx_lib_path));
    lib_unit_tests.addIncludePath(onnx_header_path_lazy);
    lib_unit_tests.addLibraryPath(onnx_lib_path_lazy);
    lib_unit_tests.linkSystemLibrary("onnxruntime");

    const run_lib_unit_tests = b.addRunArtifact(lib_unit_tests);

    const exe_unit_tests = b.addTest(.{
        .root_source_file = b.path("src/main.zig"),
        .target = target,
        .optimize = optimize,
    });

    const run_exe_unit_tests = b.addRunArtifact(exe_unit_tests);

    // Similar to creating the run step earlier, this exposes a `test` step to
    // the `zig build --help` menu, providing a way for the user to request
    // running the unit tests.
    const test_step = b.step("test", "Run unit tests");
    test_step.dependOn(&run_lib_unit_tests.step);
    test_step.dependOn(&run_exe_unit_tests.step);
}

// Although this function looks imperative, note that its job is to
// declaratively construct a build graph that will be executed by an external
// runner.
pub fn build(b: *std.Build) void {
    // Standard target options allows the person running `zig build` to choose
    // what target to build for. Here we do not override the defaults, which
    // means any target is allowed, and the default is native. Other options
    // for restricting supported target set are available.
    const target = b.standardTargetOptions(.{});

    // Standard optimization options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall. Here we do not
    // set a preferred release mode, allowing the user to decide how to optimize.
    const optimize = b.standardOptimizeOption(.{});

    const target_os = target.result.os.tag;
    const host_os = b.graph.host.result.os.tag;
    const arch = target.result.cpu.arch;

    // workaround on macos, in order set rpath correct as zig build doesn't yet support adding args to builder.
    if (target_os == .macos and arch == .aarch64 and host_os == .macos) {
        build_mac_os(b, target, optimize);
    } else {
        std.debug.panic("Unsupported OS / arch {}, {}", .{ target_os, arch });
    }
}
