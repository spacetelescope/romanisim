import os
import os.path as op
import pprint
import shutil
import sys
from difflib import unified_diff
from io import StringIO
from pathlib import Path
from typing import List

import asdf
import requests
from asdf.commands.diff import diff as asdf_diff
from ci_watson.artifactory_helpers import (
    check_url,
    get_bigdata_root,
    get_bigdata,
    BigdataError,
)
# from romancal.lib.suffix import replace_suffix
from romancal.stpipe import RomanStep

# Define location of default Artifactory API key, for Jenkins use only
ARTIFACTORY_API_KEY_FILE = '/eng/ssb2/keys/svc_rodata.key'


class RegtestData:
    """Defines data paths on Artifactory and data retrieval methods"""

    def __init__(
            self,
            env: str = "dev",
            inputs_root: str = "roman-pipeline",
            results_root: str = "roman-pipeline-results",
            docopy: bool = True,
            input: str = None,
            input_remote: str = None,
            output: str = None,
            truth: str = None,
            truth_remote: str = None,
            remote_results_path: str = None,
            test_name: str = None,
            traceback=None,
    ):
        self._env = env
        self._inputs_root = Path(inputs_root)
        self._results_root = Path(results_root)
        self._bigdata_root = Path(get_bigdata_root())

        self.docopy = docopy

        # Initialize @property attributes
        self.input = input
        self.input_remote = input_remote
        self.output = output
        self.truth = truth
        self.truth_remote = truth_remote

        # No @properties for the following attributes
        self.remote_results_path = Path(remote_results_path)
        self.test_name = test_name
        self.traceback = traceback

        # Initialize non-initialized attributes
        self.asn = None

    def __repr__(self) -> str:
        return pprint.pformat(
            dict(
                input=self.input,
                output=self.output,
                truth=self.truth,
                input_remote=self.input_remote,
                truth_remote=self.truth_remote,
                remote_results_path=self.remote_results_path,
                test_name=self.test_name,
                traceback=self.traceback,
            ),
            indent=1,
        )

    @property
    def input_remote(self) -> Path:
        return self._input_remote

    @input_remote.setter
    def input_remote(self, value: str):
        if value:
            if not isinstance(value, Path):
                value = Path(value)
        else:
            value = None
        self._input_remote = value

    @property
    def truth_remote(self) -> Path:
        return self._truth_remote

    @truth_remote.setter
    def truth_remote(self, value: str):
        if value:
            if not isinstance(value, Path):
                value = Path(value)
        else:
            value = None
        self._truth_remote = value

    @property
    def input(self) -> Path:
        return self._input

    @input.setter
    def input(self, value: str):
        if value:
            if not isinstance(value, Path):
                value = Path(value)
            value = value.absolute()
        else:
            value = None
        self._input = value

    @property
    def truth(self) -> Path:
        return self._truth

    @truth.setter
    def truth(self, value: str):
        if value:
            if not isinstance(value, Path):
                value = Path(value)
            value = value.absolute()
        else:
            value = None
        self._truth = value

    @property
    def output(self) -> Path:
        return self._output

    @output.setter
    def output(self, value: str):
        if value:
            if not isinstance(value, Path):
                value = Path(value)
            value = value.absolute()
        else:
            value = None
        self._output = value

    @property
    def bigdata_root(self) -> Path:
        return self._bigdata_root

    # The methods
    def get_data(self, path: str = None, docopy: bool = None):
        """Copy data from Artifactory remote resource to the CWD

        Updates self.input and self.input_remote upon completion
        """
        if path is None:
            path = self.input_remote
        else:
            self.input_remote = path
        if docopy is None:
            docopy = self.docopy
        self.input = get_bigdata(
            self._inputs_root,
            self._env,
            path,
            docopy=docopy,
        )
        self.input_remote = self._inputs_root / self._env / path

        return self.input

    def data_glob(self, path: str = None, glob_pattern: str = '*'):
        """Get a list of files"""
        if path is None:
            path = self.input_remote
        else:
            self.input_remote = path

        # Get full path and proceed depending on whether
        # is a local path or URL.
        root = self.bigdata_root
        if op.exists(root):
            root_path = op.join(root, self._inputs_root, self._env)
            root_len = len(root_path) + 1
            path = op.join(root_path, path)
            file_paths = _data_glob_local(path, glob_pattern)
        elif check_url(root):
            root_len = len(self._env) + 1
            file_paths = _data_glob_url(
                self._inputs_root,
                self._env, path,
                glob_pattern,
                root=root,
            )
        else:
            raise BigdataError(f'Path cannot be found: {path}')

        # Remove the root from the paths
        file_paths = [
            file_path[root_len:]
            for file_path in file_paths
        ]
        return file_paths

    def get_truth(self, path: str = None, docopy: bool = None):
        """Copy truth data from Artifactory remote resource to the CWD/truth

        Updates self.truth and self.truth_remote on completion
        """
        if path is None:
            path = self.truth_remote
        else:
            self.truth_remote = path
        if docopy is None:
            docopy = self.docopy
        os.makedirs('truth', exist_ok=True)
        os.chdir('truth')
        try:
            self.truth = get_bigdata(
                self._inputs_root,
                self._env,
                path,
                docopy=docopy,
            )
            self.truth_remote = os.path.join(self._inputs_root,
                                             self._env, path)
        except BigdataError:
            os.chdir('..')
            raise
        os.chdir('..')

        return self.truth

    # def get_asn(self, path=None, docopy=True, get_members=True):
    #     """Copy association and association members from Artifactory remote
    #     resource to the CWD/truth.
    #
    #     Updates self.input and self.input_remote upon completion
    #
    #     Parameters
    #     ----------
    #     path: str
    #         The remote path
    #
    #     docopy : bool
    #         Switch to control whether or not to copy a file
    #         into the test output directory when running the test.
    #         If you wish to open the file directly from remote
    #         location or just to set path to source, set this to `False`.
    #         Default: `True`
    #
    #     get_members: bool
    #         If an association is the input, retrieve the members.
    #         Otherwise, do not.
    #     """
    #     if path is None:
    #         path = self.input_remote
    #     else:
    #         self.input_remote = path
    #     if docopy is None:
    #         docopy = self.docopy
    #
    #     # Get the association JSON file
    #     self.input = get_bigdata(self._inputs_root, self._env, path,
    #         docopy=docopy)
    #     with open(self.input) as fp:
    #         asn = load_asn(fp)
    #         self.asn = asn
    #
    #     # Get each member in the association as well
    #     if get_members:
    #         for product in asn['products']:
    #             for member in product['members']:
    #                 fullpath = os.path.join(
    #                     os.path.dirname(self.input_remote),
    #                     member['expname'])
    #                 get_bigdata(self._inputs_root, self._env, fullpath,
    #                             docopy=self.docopy)

    def to_asdf(self, path: str):
        tree = eval(str(self))
        af = asdf.AsdfFile(tree=tree)
        af.write_to(path)

    @classmethod
    def open(cls, filename: str):
        with asdf.open(filename) as af:
            return cls(**af.tree)


def run_step_from_dict(rtdata: RegtestData, **step_params) -> RegtestData:
    """
    Run Steps with given parameter

    :param rtdata: The artifactory instance
    :param input_path: The input file path, relative to artifactory
    :param step: The step to run, either a class or a config file
    :param args: The arguments passed to `Step.from_cmdline`
    :returns rtdata: Updated `RegtestData` object with inputs set.
    """

    # Get the data. If `step_params['input_path]` is not
    # specified, the presumption is that `rtdata.input` has
    # already been retrieved.
    # input_path = step_params.get('input_path', None)
    # if input_path:
    #     try:
    #         rtdata.get_asn(input_path)
    #     except AssociationNotValidError:
    #         rtdata.get_data(input_path)

    # Figure out whether we have a config or class
    step = step_params['step']
    if step.endswith(('.asdf', '.cfg')):
        step = os.path.join('config', step)

    # Run the step
    full_args = [step, rtdata.input]
    full_args.extend(step_params['args'])

    RomanStep.from_cmdline(full_args)

    return rtdata


def run_step_from_dict_mock(rtdata: RegtestData, source: str) -> RegtestData:
    """
    Pretend to run Steps with given parameter but just copy data
    For long-running steps where the result already exists, just copy the data from source

    :param rtdata: The artifactory instance
    :param input_path: The input file path, relative to artifactory
    :param step: The step to run, either a class or a config file
    :param args: The arguments passed to `Step.from_cmdline`
    :param source: The folder to copy from. All regular files are copied.
    :returns rtdata: Updated `RegtestData` object with inputs set.
    """

    # Get the data. If `step_params['input_path]` is not
    # specified, the presumption is that `rtdata.input` has
    # already been retrieved.
    # input_path = step_params.get('input_path', None)
    # if input_path:
    #     try:
    #         rtdata.get_asn(input_path)
    #     except AssociationNotValidError:
    #         rtdata.get_data(input_path)

    # Copy the data
    for file_name in os.listdir(source):
        file_path = os.path.join(source, file_name)
        if os.path.isfile(file_path):
            shutil.copy(file_path, '.')

    return rtdata


def is_like_truth(
        rtdata: RegtestData,
        ignore_asdf_paths: bool,
        output: str,
        truth_path: str,
        is_suffix: bool = True,
):
    """
    Compare step outputs with truth

    :param rtdata: The artifactory object from the step run.
    :param ignore_asdf_paths: The asdf `diff` keyword arguments
    :param output: The suffix or full file name to check on.
    :param truth_path: Location of the truth files.
    :param is_suffix: Interpret `output` as just a suffix on the expected output root. Otherwise, assume it is a full file name
    """
    __tracebackhide__ = True
    # If given only a suffix, get the root to change the suffix of.
    # If the input was an association, the output should be the name of
    # the product. Otherwise, output is based on input.
    if is_suffix:
        # suffix = output
        if rtdata.asn:
            output = rtdata.asn['products'][0]['name']
        else:
            output = os.path.splitext(os.path.basename(rtdata.input))[0]
        # output = replace_suffix(output, suffix) + '.asdf'
    rtdata.output = output

    rtdata.get_truth(os.path.join(truth_path, output))

    # diff = FITSDiff(rtdata.output, rtdata.truth, **fitsdiff_default_kwargs)
    report = compare_asdf(rtdata.output, rtdata.truth, **ignore_asdf_paths)
    assert report is None, report


def text_diff(from_path: str, to_path: str) -> List[str]:
    """
    Assertion helper for diffing two text files

    :param from_path: File to diff from.
    :param to_path: File to diff to.  The truth.
    :returns diffs: A generator of a list of strings that are the differences. The output from `difflib.unified_diff`
    """
    __tracebackhide__ = True
    with open(from_path) as fh:
        from_lines = fh.readlines()
    with open(to_path) as fh:
        to_lines = fh.readlines()

    diffs = unified_diff(from_lines, to_lines, from_path, to_path)

    diff = list(diffs)
    if len(diff) > 0:
        diff.insert(0, "\n")
        raise AssertionError("".join(diff))
    else:
        return diff


def _data_glob_local(*glob_pattern_parts) -> List[Path]:
    """
    Perform a glob on the local path

    :param glob_pattern_parts: List of components that will be built into a single path
    :returns file_paths: Full file paths that match the glob criterion
    """
    full_glob = Path().joinpath(*glob_pattern_parts)
    return Path().glob(str(full_glob))


def _data_glob_url(*url_parts, root: str = None) -> str:
    """
    :param url: List of components that will be used to create a URL path
    :param root: The root server path to the Artifactory server. Normally retrieved from `get_bigdata_root`.
    :returns: Full URLS that match the glob criterion
    """
    # Fix root root-ed-ness
    if not isinstance(root, Path):
        root = Path(root)

    # Access
    try:
        envkey = os.environ['API_KEY_FILE']
    except KeyError:
        envkey = ARTIFACTORY_API_KEY_FILE

    try:
        with open(envkey) as fp:
            headers = {'X-JFrog-Art-Api': fp.readline().strip()}
    except (PermissionError, FileNotFoundError):
        print(
            "Warning: Anonymous Artifactory search requests are limited to "
            "1000 results. Use an API key and define API_KEY_FILE environment"
            "variable to get full search results.",
            file=sys.stderr,
        )
        headers = None

    search_url = root / 'api/search/pattern'

    # Join and re-split the url so that every component is identified.
    url = root.joinpath(*url_parts)
    all_parts = url.parts

    # Pick out "roman-pipeline", the repo name
    repo = all_parts[4]

    # Format the pattern
    pattern = f'{repo}:{Path().joinpath(*all_parts[5:])}'

    # Make the query
    params = {'pattern': pattern}
    with requests.get(str(search_url), params=params, headers=headers) as r:
        url_paths = r.json()['files']

    return url_paths


def compare_asdf(result: str, truth: str, **kwargs):
    with StringIO() as f:
        asdf_diff(
            [result, truth],
            minimal=False,
            iostream=f,
            **kwargs,
        )
        if f.getvalue():
            f.getvalue()
