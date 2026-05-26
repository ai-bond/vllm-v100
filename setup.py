# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path
from shutil import which

import torch
from packaging.version import Version, parse
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools_scm import get_version
from torch.utils.cpp_extension import CUDA_HOME


def load_module_from_path(module_name, path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


ROOT_DIR = Path(__file__).parent
logger = logging.getLogger(__name__)

envs = load_module_from_path("envs", os.path.join(ROOT_DIR, "vllm", "envs.py"))

if not sys.platform.startswith("linux"):
    logger.warning(
        "vLLM CUDA build only supports Linux (including WSL). "
        "Building on %s, so vLLM may not run correctly.",
        sys.platform,
    )

VLLM_TARGET_DEVICE = "cuda"

if torch.version.cuda is None:
    raise RuntimeError(
        "CUDA-only build requires a CUDA-enabled PyTorch. "
        "Please install PyTorch with CUDA 12.9 support first:\n"
        "  pip install torch==2.10.0+cu129 "
        "--index-url https://download.pytorch.org/whl/cu129"
    )

logger.info("Building vLLM-v100 fork for CUDA only (Volta V100, SM 7.0)")


def is_sccache_available() -> bool:
    return which("sccache") is not None and not bool(
        int(os.getenv("VLLM_DISABLE_SCCACHE", "0"))
    )


def is_ccache_available() -> bool:
    return which("ccache") is not None


def is_ninja_available() -> bool:
    return which("ninja") is not None


def is_freethreaded():
    return bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


class CMakeExtension(Extension):
    def __init__(self, name: str, cmake_lists_dir: str = ".", **kwa) -> None:
        super().__init__(
            name,
            sources=[],
            py_limited_api=not is_freethreaded(),
            **kwa,
        )
        self.cmake_lists_dir = os.path.abspath(cmake_lists_dir)


class cmake_build_ext(build_ext):
    did_config: dict[str, bool] = {}

    def compute_num_jobs(self):
        num_jobs = envs.MAX_JOBS
        if num_jobs is not None:
            num_jobs = int(num_jobs)
            logger.info("Using MAX_JOBS=%d as the number of jobs.", num_jobs)
        else:
            try:
                num_jobs = len(os.sched_getaffinity(0))
            except AttributeError:
                num_jobs = os.cpu_count()

        nvcc_threads = None
        if CUDA_HOME is not None:
            try:
                nvcc_version = get_nvcc_cuda_version()
                if nvcc_version >= Version("11.2"):
                    nvcc_threads = envs.NVCC_THREADS
                    if nvcc_threads is not None:
                        nvcc_threads = int(nvcc_threads)
                        logger.info(
                            "Using NVCC_THREADS=%d as the number of nvcc threads.",
                            nvcc_threads,
                        )
                    else:
                        nvcc_threads = 1
                    num_jobs = max(1, num_jobs // nvcc_threads)
            except Exception as e:
                logger.warning("Failed to get NVCC version: %s", e)

        return num_jobs, nvcc_threads

    def configure(self, ext: CMakeExtension) -> None:
        if ext.cmake_lists_dir in cmake_build_ext.did_config:
            return

        cmake_build_ext.did_config[ext.cmake_lists_dir] = True

        default_cfg = "Debug" if self.debug else "RelWithDebInfo"
        cfg = envs.CMAKE_BUILD_TYPE or default_cfg

        cmake_args = [
            "-DCMAKE_BUILD_TYPE={}".format(cfg),
            "-DVLLM_TARGET_DEVICE={}".format(VLLM_TARGET_DEVICE),
        ]

        verbose = envs.VERBOSE
        if verbose:
            cmake_args += ["-DCMAKE_VERBOSE_MAKEFILE=ON"]

        if is_sccache_available():
            cmake_args += [
                "-DCMAKE_C_COMPILER_LAUNCHER=sccache",
                "-DCMAKE_CXX_COMPILER_LAUNCHER=sccache",
                "-DCMAKE_CUDA_COMPILER_LAUNCHER=sccache",
            ]
        elif is_ccache_available():
            cmake_args += [
                "-DCMAKE_C_COMPILER_LAUNCHER=ccache",
                "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache",
                "-DCMAKE_CUDA_COMPILER_LAUNCHER=ccache",
            ]

        cmake_args += ["-DVLLM_PYTHON_EXECUTABLE={}".format(sys.executable)]
        cmake_args += ["-DVLLM_PYTHON_PATH={}".format(":".join(sys.path))]

        fc_base_dir = os.path.join(ROOT_DIR, ".deps")
        fc_base_dir = os.environ.get("FETCHCONTENT_BASE_DIR", fc_base_dir)
        cmake_args += ["-DFETCHCONTENT_BASE_DIR={}".format(fc_base_dir)]

        num_jobs, nvcc_threads = self.compute_num_jobs()

        if nvcc_threads:
            cmake_args += ["-DNVCC_THREADS={}".format(nvcc_threads)]

        if is_ninja_available():
            build_tool = ["-G", "Ninja"]
            cmake_args += [
                "-DCMAKE_JOB_POOL_COMPILE:STRING=compile",
                "-DCMAKE_JOB_POOLS:STRING=compile={}".format(num_jobs),
            ]
        else:
            build_tool = []

        assert CUDA_HOME is not None, (
            "CUDA_HOME is not set. Please install CUDA Toolkit and set CUDA_HOME."
        )
        cmake_args += [f"-DCMAKE_CUDA_COMPILER={CUDA_HOME}/bin/nvcc"]

        other_cmake_args = os.environ.get("CMAKE_ARGS")
        if other_cmake_args:
            cmake_args += other_cmake_args.split()

        subprocess.check_call(
            ["cmake", ext.cmake_lists_dir, *build_tool, *cmake_args],
            cwd=self.build_temp,
        )

    def build_extensions(self) -> None:
        try:
            subprocess.check_output(["cmake", "--version"])
        except OSError as e:
            raise RuntimeError("Cannot find CMake executable") from e

        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        targets = []

        def target_name(s: str) -> str:
            return s.removeprefix("vllm.")

        for ext in self.extensions:
            self.configure(ext)
            targets.append(target_name(ext.name))

        num_jobs, _ = self.compute_num_jobs()

        build_args = [
            "--build",
            ".",
            f"-j={num_jobs}",
            *[f"--target={name}" for name in targets],
        ]

        subprocess.check_call(["cmake", *build_args], cwd=self.build_temp)

        for ext in self.extensions:
            outdir = Path(self.get_ext_fullpath(ext.name)).parent.absolute()

            if outdir == self.build_temp:
                continue

            prefix = outdir
            for _ in range(ext.name.count(".")):
                prefix = prefix.parent

            install_args = [
                "cmake",
                "--install",
                ".",
                "--prefix",
                prefix,
                "--component",
                target_name(ext.name),
            ]
            subprocess.check_call(install_args, cwd=self.build_temp)

    def run(self):
        super().run()

        print(
            f"Copying {self.build_lib}/vllm/third_party/triton_kernels "
            "to vllm/third_party/triton_kernels"
        )
        shutil.copytree(
            f"{self.build_lib}/vllm/third_party/triton_kernels",
            "vllm/third_party/triton_kernels",
            dirs_exist_ok=True,
        )


class precompiled_build_ext(build_ext):
    """Disables extension building when using precompiled binaries."""

    def run(self) -> None:
        return

    def build_extensions(self) -> None:
        print("Skipping build_ext: using precompiled extensions.")
        return


class precompiled_wheel_utils:
    """Extracts libraries and other files from an existing wheel."""

    @staticmethod
    def fetch_metadata_for_variant(
        commit: str, variant: str | None
    ) -> tuple[list[dict], str]:
        variant_dir = f"{variant}/" if variant is not None else ""
        repo_url = f"https://wheels.vllm.ai/{commit}/{variant_dir}vllm/"
        meta_url = repo_url + "metadata.json"
        print(f"Trying to fetch nightly build metadata from {meta_url}")
        from urllib.request import urlopen

        with urlopen(meta_url) as resp:
            wheels = json.loads(resp.read().decode("utf-8"))
        return wheels, repo_url

    @staticmethod
    def detect_system_cuda_variant() -> str:
        """Auto-detect CUDA variant from torch, nvidia-smi, or env default."""
        supported = {12: "cu129", 13: "cu130"}

        if envs.is_set("VLLM_MAIN_CUDA_VERSION"):
            v = envs.VLLM_MAIN_CUDA_VERSION
            print(f"Using VLLM_MAIN_CUDA_VERSION={v}")
            return "cu" + v.replace(".", "")[:3]

        cuda_version = None
        try:
            import torch
            cuda_version = torch.version.cuda
        except Exception:
            pass

        if not cuda_version:
            try:
                out = subprocess.run(
                    ["nvidia-smi"], capture_output=True, text=True, timeout=10
                )
                if m := re.search(r"CUDA Version:\s*(\d+\.\d+)", out.stdout):
                    cuda_version = m.group(1)
            except Exception:
                pass

        if not cuda_version:
            cuda_version = envs.VLLM_MAIN_CUDA_VERSION

        major = int(cuda_version.split(".")[0])
        variant = supported.get(major, supported[max(supported)])
        print(f"Detected CUDA {cuda_version}, using variant {variant}")
        return variant

    @staticmethod
    def fetch_wheel_from_pypi_index(index_url: str, package: str = "vllm") -> str:
        import platform
        from html.parser import HTMLParser
        from urllib.parse import urljoin
        from urllib.request import urlopen

        arch = platform.machine()

        class WheelLinkParser(HTMLParser):
            def __init__(self):
                super().__init__()
                self.wheels = []

            def handle_starttag(self, tag, attrs):
                if tag == "a":
                    for name, value in attrs:
                        if name == "href" and value.endswith(".whl"):
                            self.wheels.append(value)

        simple_url = f"{index_url.rstrip('/')}/{package}/"
        print(f"Fetching wheel list from {simple_url}")
        with urlopen(simple_url) as resp:
            html = resp.read().decode("utf-8")

        parser = WheelLinkParser()
        parser.feed(html)

        for wheel in reversed(parser.wheels):
            if arch in wheel:
                if wheel.startswith("http"):
                    return wheel
                return urljoin(simple_url, wheel)

        raise ValueError(f"No compatible wheel found for {arch} at {simple_url}")

    @staticmethod
    def determine_wheel_url() -> tuple[str, str | None]:
        """Try to determine the precompiled wheel URL or path to use."""
        wheel_location = os.getenv("VLLM_PRECOMPILED_WHEEL_LOCATION", None)
        if wheel_location is not None:
            print(f"Using user-specified precompiled wheel location: {wheel_location}")
            return wheel_location, None

        import platform
        arch = platform.machine()

        variant = os.getenv("VLLM_PRECOMPILED_WHEEL_VARIANT", None)
        if variant is None:
            variant = precompiled_wheel_utils.detect_system_cuda_variant()

        commit = os.getenv("VLLM_PRECOMPILED_WHEEL_COMMIT", "").lower()
        if not commit or len(commit) != 40:
            print(
                f"VLLM_PRECOMPILED_WHEEL_COMMIT not valid: {commit}, "
                "trying to fetch base commit in main branch"
            )
            commit = precompiled_wheel_utils.get_base_commit_in_main_branch()

        print(f"Using precompiled wheel commit {commit} with variant {variant}")

        try_default = False
        wheels, repo_url, download_filename = None, None, None
        try:
            wheels, repo_url = precompiled_wheel_utils.fetch_metadata_for_variant(
                commit, variant
            )
        except Exception as e:
            logger.warning(
                "Failed to fetch precompiled wheel metadata for variant %s: %s",
                variant,
                e,
            )
            try_default = True

        if try_default:
            print("Trying the default variant from remote")
            wheels, repo_url = precompiled_wheel_utils.fetch_metadata_for_variant(
                commit, None
            )

        assert wheels is not None and repo_url is not None, (
            "Failed to fetch precompiled wheel metadata"
        )

        from urllib.parse import urljoin

        for wheel in wheels:
            if wheel.get("package_name") == "vllm" and arch in wheel.get(
                "platform_tag", ""
            ):
                print(f"Found precompiled wheel metadata: {wheel}")
                if "path" not in wheel:
                    raise ValueError(f"Wheel metadata missing path: {wheel}")
                wheel_url = urljoin(repo_url, wheel["path"])
                download_filename = wheel.get("filename")
                print(f"Using precompiled wheel URL: {wheel_url}")
                break
        else:
            raise ValueError(
                f"No precompiled vllm wheel found for architecture {arch} "
                f"from repo {repo_url}. All available wheels: {wheels}"
            )

        return wheel_url, download_filename

    @staticmethod
    def extract_precompiled_and_patch_package(
        wheel_url_or_path: str, download_filename: str | None
    ) -> dict:
        import tempfile
        import zipfile

        temp_dir = None
        try:
            if not os.path.isfile(wheel_url_or_path):
                wheel_filename = download_filename or wheel_url_or_path.split("/")[-1]
                temp_dir = tempfile.mkdtemp(prefix="vllm-wheels")
                wheel_path = os.path.join(temp_dir, wheel_filename)
                print(f"Downloading wheel from {wheel_url_or_path} to {wheel_path}")
                from urllib.request import urlretrieve

                urlretrieve(wheel_url_or_path, filename=wheel_path)
            else:
                wheel_path = wheel_url_or_path
                print(f"Using existing wheel at {wheel_path}")

            package_data_patch = {}

            with zipfile.ZipFile(wheel_path) as wheel:
                files_to_copy = [
                    "vllm/_C.abi3.so",
                    "vllm/_C_stable_libtorch.abi3.so",
                    "vllm/_moe_C.abi3.so",
                    "vllm/cumem_allocator.abi3.so",
                ]

                triton_kernels_regex = re.compile(
                    r"vllm/third_party/triton_kernels/(?:[^/.][^/]*/)*(?!\.)[^/]*\.py"
                )

                file_members = list(
                    filter(lambda x: x.filename in files_to_copy, wheel.filelist)
                )
                file_members += list(
                    filter(
                        lambda x: triton_kernels_regex.match(x.filename),
                        wheel.filelist,
                    )
                )

                for file in file_members:
                    print(f"[extract] {file.filename}")
                    target_path = os.path.join(".", file.filename)
                    os.makedirs(os.path.dirname(target_path), exist_ok=True)
                    with (
                        wheel.open(file.filename) as src,
                        open(target_path, "wb") as dst,
                    ):
                        shutil.copyfileobj(src, dst)

                    pkg = os.path.dirname(file.filename).replace("/", ".")
                    package_data_patch.setdefault(pkg, []).append(
                        os.path.basename(file.filename)
                    )

            return package_data_patch
        finally:
            if temp_dir is not None:
                print(f"Removing temporary directory {temp_dir}")
                shutil.rmtree(temp_dir)

    @staticmethod
    def get_base_commit_in_main_branch() -> str:
        try:
            # Point to the fork repository
            curl_cmd = [
                "curl",
                "-s",
                "https://api.github.com/repos/ai-bond/vllm-v100/commits/main",
            ]
            github_token = os.getenv("GH_TOKEN", os.getenv("GITHUB_TOKEN"))
            if github_token:
                curl_cmd += ["-H", f"Authorization: token {github_token}"]
            resp_json = subprocess.check_output(curl_cmd).decode("utf-8")
            upstream_main_commit = json.loads(resp_json)["sha"]
            print(f"Upstream main branch latest commit: {upstream_main_commit}")

            if envs.VLLM_DOCKER_BUILD_CONTEXT:
                return upstream_main_commit

            try:
                subprocess.check_output(
                    ["git", "cat-file", "-e", f"{upstream_main_commit}"]
                )
            except subprocess.CalledProcessError:
                subprocess.check_call(
                    ["git", "fetch", "https://github.com/ai-bond/vllm-v100", "main"]
                )

            current_branch = (
                subprocess.check_output(["git", "branch", "--show-current"])
                .decode("utf-8")
                .strip()
            )

            base_commit = (
                subprocess.check_output(
                    ["git", "merge-base", f"{upstream_main_commit}", current_branch]
                )
                .decode("utf-8")
                .strip()
            )
            return base_commit
        except ValueError as err:
            raise ValueError(err) from None
        except Exception as err:
            logger.warning(
                "Failed to get the base commit in the main branch. "
                "Using the nightly wheel. The libraries in this "
                "wheel may not be compatible with your dev branch: %s",
                err,
            )
            return "nightly"


def get_nvcc_cuda_version() -> Version:
    """Get the CUDA version from nvcc."""
    assert CUDA_HOME is not None, "CUDA_HOME is not set"
    nvcc_output = subprocess.check_output(
        [CUDA_HOME + "/bin/nvcc", "-V"], universal_newlines=True
    )
    output = nvcc_output.split()
    release_idx = output.index("release") + 1
    nvcc_cuda_version = parse(output[release_idx].split(",")[0])
    return nvcc_cuda_version


def get_vllm_version() -> str:
    if env_version := os.getenv("VLLM_VERSION_OVERRIDE"):
        print(f"Overriding VLLM version with {env_version} from VLLM_VERSION_OVERRIDE")
        os.environ["SETUPTOOLS_SCM_PRETEND_VERSION"] = env_version
        return get_version(write_to="vllm/_version.py")

    version = get_version(write_to="vllm/_version.py")
    sep = "+" if "+" not in version else "."

    if envs.VLLM_USE_PRECOMPILED and not envs.VLLM_SKIP_PRECOMPILED_VERSION_SUFFIX:
        version += f"{sep}precompiled"
    else:
        cuda_version = str(get_nvcc_cuda_version())
        if cuda_version != envs.VLLM_MAIN_CUDA_VERSION:
            cuda_version_str = cuda_version.replace(".", "")[:3]
            if "sdist" not in sys.argv:
                version += f"{sep}cu{cuda_version_str}"

    # Add v100 suffix for Volta fork identification
    version += f"{sep}v100"

    return version


def get_requirements() -> list[str]:
    """Get Python package dependencies from requirements.txt."""
    requirements_dir = ROOT_DIR / "requirements"

    def _read_requirements(filename: str) -> list[str]:
        with open(requirements_dir / filename) as f:
            requirements = f.read().strip().split("\n")
        resolved_requirements = []
        for line in requirements:
            if line.startswith("-r "):
                resolved_requirements += _read_requirements(line.split()[1])
            elif (
                not line.startswith("--")
                and not line.startswith("#")
                and line.strip() != ""
            ):
                resolved_requirements.append(line)
        return resolved_requirements

    requirements = _read_requirements("cuda.txt")
    modified_requirements = []
    for req in requirements:
        if "vllm-flash-attn" in req or "flash-attn" in req:
            continue
        if "xformers" in req:
            continue
        modified_requirements.append(req)
    return modified_requirements

ext_modules = []

ext_modules.append(CMakeExtension(name="vllm._moe_C"))
ext_modules.append(CMakeExtension(name="vllm.cumem_allocator"))
ext_modules.append(CMakeExtension(name="vllm.triton_kernels", optional=True))
ext_modules.append(CMakeExtension(name="vllm._C"))
ext_modules.append(CMakeExtension(name="vllm._C_stable_libtorch"))

package_data = {
    "vllm": [
        "py.typed",
        "libs/*.so",
        "model_executor/layers/fused_moe/configs/*.json",
        "model_executor/layers/quantization/utils/configs/*.json",
        "entrypoints/serve/instrumentator/static/*.js",
        "entrypoints/serve/instrumentator/static/*.css",
    ]
}

if envs.VLLM_USE_PRECOMPILED:
    wheel_url, download_filename = precompiled_wheel_utils.determine_wheel_url()
    patch = precompiled_wheel_utils.extract_precompiled_and_patch_package(
        wheel_url, download_filename
    )
    for pkg, files in patch.items():
        package_data.setdefault(pkg, []).extend(files)

cmdclass = {
    "build_ext": (
        precompiled_build_ext if envs.VLLM_USE_PRECOMPILED else cmake_build_ext
    ),
}


setup(
    version=get_vllm_version(),
    ext_modules=ext_modules,
    install_requires=get_requirements(),
    extras_require={
        "bench": ["pandas", "matplotlib", "seaborn", "datasets", "scipy", "plotly"],
        "tensorizer": ["tensorizer==2.10.1"],
        "fastsafetensors": ["fastsafetensors >= 0.2.2"],
        "instanttensor": ["instanttensor >= 0.1.5"],
        "runai": ["runai-model-streamer[s3,gcs,azure] >= 0.15.7"],
        "audio": [
            "av",
            "resampy",
            "scipy",
            "soundfile",
            "mistral_common[audio]",
        ],
        "video": [],
        "flashinfer": [],
        "grpc": ["smg-grpc-servicer[vllm] >= 0.5.0"],
        "otel": [
            "opentelemetry-sdk >=1.26.0",
            "opentelemetry-api >=1.26.0",
            "opentelemetry-exporter-otlp >=1.26.0",
            "opentelemetry-semantic-conventions-ai >=0.4.1",
        ],
    },
    cmdclass=cmdclass,
    package_data=package_data,
)
