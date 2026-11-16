"""Ship the compiled-core sources in the package; mark wheels that carry
ctypes-loaded native kernels as platform-specific."""

from pathlib import Path

from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_py import build_py

ROOT = Path(__file__).resolve().parent
CORE_SOURCES = (".cpp", ".cu", ".h", ".txt", ".sh")


class IterCCBuild(build_py):
    """Copy core/ (without build trees) to itercc/_core_src, so an installed
    package compiles its kernels on first use (`_cpu_kernels`) instead of
    falling back to NumPy when no prebuilt library was installed."""

    def run(self):
        super().run()
        core = ROOT / "core"
        if not (core / "include" / "itercc" / "core.h").exists():
            return
        target = Path(self.build_lib) / "itercc" / "_core_src"
        for path in sorted(core.rglob("*")):
            relative = path.relative_to(core)
            if (path.is_file() and path.suffix in CORE_SOURCES
                    and not relative.parts[0].startswith("build")):
                (target / relative).parent.mkdir(parents=True, exist_ok=True)
                self.copy_file(str(path), str(target / relative))


class IterCCWheel(bdist_wheel):
    def finalize_options(self):
        super().finalize_options()
        package = ROOT / "itercc"
        self.root_is_pure = not any(
            path.name != "libitercc_core.native.so" for path in package.glob("libitercc_*.so")
        )

    def get_tag(self):
        python, abi, platform = super().get_tag()
        if not self.root_is_pure:
            # The libraries use ctypes and have no Python binary ABI dependency.
            return "py3", "none", platform
        return python, abi, platform


setup(cmdclass={"build_py": IterCCBuild, "bdist_wheel": IterCCWheel})
