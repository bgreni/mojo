# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #


from mblib2to3.pgen2.driver import load_packaged_grammar

# Two grammars that differ by one rule. pgen compiles unreferenced rules too,
# so `extra_stmt` in the symbol table is a reliable marker for which of the
# two a loaded grammar was built from.
GRAMMAR = "file_input: (NEWLINE | stmt)* ENDMARKER\nstmt: NAME NEWLINE\n"
GRAMMAR_WITH_EXTRA_RULE = GRAMMAR + "extra_stmt: 'extra' NEWLINE\n"


def _load(grammar_path, cache_dir):
    return load_packaged_grammar("mblib2to3", str(grammar_path), cache_dir)


def test_pickle_is_not_shared_by_installs_with_different_grammars(tmp_path):
    # Two toolchains installed side by side, each with its own Grammar.txt,
    # both caching into one directory -- the default, since the cache path is
    # keyed on mblack's version string, which is identical in every build.
    # Both grammar files are named Grammar.txt, so they collapse onto a single
    # pickle name, and whichever ran last wins for both.
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()

    other_install = tmp_path / "other_install"
    other_install.mkdir()
    other_grammar = other_install / "Grammar.txt"
    other_grammar.write_text(GRAMMAR_WITH_EXTRA_RULE)

    this_install = tmp_path / "this_install"
    this_install.mkdir()
    this_grammar = this_install / "Grammar.txt"
    this_grammar.write_text(GRAMMAR)

    # This install runs first and leaves its grammar in the shared cache. The
    # pickle is now newer than the other install's Grammar.txt, which was
    # written when that toolchain was unpacked and does not change again.
    _load(this_grammar, cache_dir)

    grammar = _load(other_grammar, cache_dir)

    assert "extra_stmt" in grammar.symbol2number


def test_pickle_is_reused_when_the_grammar_is_unchanged(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    grammar_path = tmp_path / "Grammar.txt"
    grammar_path.write_text(GRAMMAR)

    _load(grammar_path, cache_dir)
    cached = list(cache_dir.iterdir())
    assert len(cached) == 1
    written_at = cached[0].stat().st_mtime_ns

    _load(grammar_path, cache_dir)

    assert [path.name for path in cache_dir.iterdir()] == [cached[0].name]
    assert cached[0].stat().st_mtime_ns == written_at
