import os.path
import re
import sys
import traceback

assert __name__ == "__main__", "run as [%s] ?!" % __name__

# set this to True then no `__all__ = [...]` will be present thus every bits get exported
wild_exports = False

# set this to True then new file contents will just be printed than actually written
dry_run = False

non_py_char = re.compile(r"[^a-zA-Z0-9\_\.]")


def parse_exports(py_file_path: str):
    exp_lines = []

    def eval_exports():
        exp_code = "\n".join(exp_lines)
        try:
            exp_def = {}
            exec(compile(exp_code, "exports: " + py_file_path, "exec"), {}, exp_def)
            return exp_def["__all__"]
        except Exception as exc:
            raise SyntaxError(
                f"Unexpected module exports code: \n\n{exp_code}\n\n"
            ) from exc

    exp_end_tok = None
    with open(py_file_path, encoding="utf-8") as pyf:
        for line in pyf:
            if exp_end_tok is None:
                # exports definition not started yet
                if line.startswith("__all__"):
                    # start of exports definition
                    try:
                        assign_pos = line.index("=")
                        exp_begin_tok = line[assign_pos + 1 :].lstrip()[0]
                        if "[" == exp_begin_tok:
                            exp_end_tok = "]"
                        elif "(" == exp_begin_tok:
                            exp_end_tok = ")"
                        else:
                            raise SyntaxError(
                                f'__all__ definition started with "{exp_begin_tok}" not understood!'
                            )
                    except ValueError:
                        # = absent
                        raise SyntaxError(
                            "Not assigning to __all__ for exports ?!"
                        ) from None
            if exp_end_tok is not None:
                # this line is part of exports definition

                try:
                    line = line.rstrip()  # trim white spaces at end anyway
                    line = line[: line.index("#")]  # trim comments if present
                    line = line.rstrip()  # trim white spaces before comments
                except ValueError:
                    # no comment
                    pass

                exp_lines.append(line)

                if line.endswith(exp_end_tok):
                    # finish of exports definition
                    return eval_exports()

    if len(exp_lines) > 0:
        raise SyntaxError("Unfinished __all__ assignment ?!")

    # no __all__ assignment at all
    return None


def gather_exports(root_dir, pkg_path, sn2exps, pkg_only=False):
    parent_dir = os.path.join(root_dir, *pkg_path)
    print(
        " *** scanning %s under dir: %s"
        % ("sub-packages" if pkg_only else "all modules", parent_dir),
        file=sys.stderr,
    )
    for sub_name in os.listdir(parent_dir):

        if "__pycache__" == sub_name:
            # ignore it of coz
            continue
        if sub_name.startswith("."):
            # ignore what ever starts with .
            continue
        # if non_py_char.search(sub_name):
        #     # ignore path contains chars other than alphabet, digits, dot, and underscore
        #     continue
        if sub_name.endswith("_modules"):
            # skip `edh_modules` `python_modules` `node_modules` etc.
            continue

        if sub_name.endswith(".py") and os.path.isfile(
            os.path.join(parent_dir, sub_name)
        ):
            # process sub-module

            if sub_name.startswith("_"):
                # ignore file module starts with _
                continue

            if pkg_only:  # ignore file module in this case
                continue

            sub_modu_path = pkg_path + [sub_name[:-3]]
            modu_abs_name = ".".join(sub_modu_path)

            if wild_exports:
                sn2exps[sub_modu_path[-1]] = True, True
                continue

            sub_modu_file_path = os.path.join(*pkg_path, sub_name)
            print(
                f" *** parsing exports of module {modu_abs_name} from file: {sub_modu_file_path} ...",
                file=sys.stderr,
            )
            try:
                exports = parse_exports(sub_modu_file_path)
                if exports is None:
                    # if __all__ is not defined, respect no export
                    exports = ()
                sn2exps[sub_modu_path[-1]] = True, tuple(exports)
            except BaseException as exc:
                traceback.print_exc(file=sys.stderr)
                print(
                    f" ^^^ above error occurred in parsing exports of {modu_abs_name}"
                    f" from module file: {sub_modu_file_path}",
                    file=sys.stderr,
                )
                sn2exps[sub_modu_path[-1]] = False, exc

        elif os.path.isdir(os.path.join(parent_dir, sub_name)):
            # process sub-package

            if os.path.islink(os.path.join(parent_dir, sub_name)):
                # but don't follow sym links
                continue

            sub_pkg_path = pkg_path + [sub_name]
            pkg_dir = os.path.join(root_dir, *sub_pkg_path)
            init_path = os.path.join(pkg_dir, "__init__.py")
            is_ns_pkg = not os.path.isfile(init_path)
            sub_exps = {}
            if is_ns_pkg:
                # no up-propagation of exports across namespace or internal (named with leading underscore) packages
                gather_exports(root_dir, sub_pkg_path, sub_exps, pkg_only=True)
                sn2exps[sub_name] = False, sub_exps
            elif sub_name.startswith("_"):
                # no up-propagation of exports from internal sub-package
                gather_exports(root_dir, sub_pkg_path, sub_exps, pkg_only=False)
                sn2exps[sub_name] = False, sub_exps
            else:
                # up-propagate exports for sub-packages with __init__.py defined
                gather_exports(root_dir, sub_pkg_path, sub_exps, pkg_only=False)
                sn2exps[sub_name] = True, sub_exps


def flat_defs(sub_data):
    bubble_up, sub_exps = sub_data
    if not bubble_up:
        return
    if isinstance(sub_exps, dict):
        for k, v in sorted(sub_exps.items()):
            yield from flat_defs(v)
    elif isinstance(sub_exps, tuple):
        yield from sub_exps
    elif isinstance(sub_exps, BaseException):
        # intermediate error, have to ignore here
        pass
    else:
        assert False, "unexpected type %r" % (type(sub_exps))


def wrapping_lines(frags, indent="", sep="", columns=79):
    line = ""
    for frag in frags:
        if len(line) <= 0:
            line = indent + frag
        elif len(line) + len(sep) + len(frag) > columns:
            yield line + "\n"
            line = indent + frag
        else:
            line += sep + frag
    yield line + "\n"


def gen_init(f, prelude, sn2exps):
    sn2flat = {sn: tuple(flat_defs(sub_data)) for sn, sub_data in sn2exps.items()}
    non_empty_exps = [(sn, exps) for sn, exps in sorted(sn2flat.items()) if exps]

    if prelude is not None:
        f.write(prelude)

    if not wild_exports:

        f.write("\n__all__ = [\n")

        for sn, exps in non_empty_exps:
            f.write("\n    # exports from .%s\n" % (sn,))
            f.writelines(
                wrapping_lines(
                    ("'%s'," % (expn,) for expn in exps), indent="    ", sep=" "
                )
            )

        for sn, (bubble_up, sub_exps) in sorted(sn2exps.items()):
            if isinstance(sub_exps, BaseException):
                f.write("\n    # .%s has error: {%s}\n" % (sn, sub_exps))

        f.write("\n]\n\n")

    f.writelines("from .%s import *\n" % (sn,) for sn, exps in non_empty_exps)


def write_out(root_dir, pkg_path, sn2exps, dry_run=True):
    pkg_dir = os.path.join(root_dir, *pkg_path)
    init_path = os.path.join(pkg_dir, "__init__.py")
    print(" *** writing to dir: " + pkg_dir, file=sys.stderr)
    if os.path.isfile(init_path):
        prelude = read_pkg_docstr(init_path)
        if dry_run:
            print(
                " vvv below contents are supposed to overwrite %s" % (init_path,),
                file=sys.stdout,
            )
            gen_init(sys.stdout, prelude, sn2exps)
            print(
                " ^^^ above contents are supposed to overwrite %s" % (init_path,),
                file=sys.stdout,
            )
        else:
            with open(init_path, "w") as f:
                gen_init(f, prelude, sn2exps)
    else:
        print(
            " !!! %s is considered a namespace package thus no __init__.py will be generated for it."
            % (pkg_dir,),
            file=sys.stderr,
        )
        print(" &&& If it is desired, touch %s" % (init_path,), file=sys.stderr)

    for sub_name, (bubble_up, sub_exps) in sn2exps.items():
        if isinstance(sub_exps, dict):
            write_out(root_dir, pkg_path + [sub_name], sub_exps, dry_run=dry_run)


def read_pkg_docstr(file_name):
    doc_lines = None
    with open(file_name, "r") as f:
        for line in f:
            if line.startswith('"""'):
                if doc_lines is None:
                    # begin of docstring
                    doc_lines = []
                else:
                    # end of docstring
                    doc_lines = tuple(doc_lines)
                    break
            elif doc_lines is not None:
                # middle of docstring
                doc_lines.append(line)
    if doc_lines is None:
        return None
    elif not isinstance(doc_lines, tuple):
        raise RuntimeError(f"File [{file_name}] has invalid docstring!")
    return '"""\n' + ("".join(doc_lines)) + '"""\n'


root_dir = os.getcwd()
print(" *** starting from root dir: " + root_dir, file=sys.stderr)
root_pkgs = {}
gather_exports(root_dir, [], root_pkgs, pkg_only=False)

write_out(root_dir, [], root_pkgs, dry_run=dry_run)

print(" *** done for root dir: " + root_dir, file=sys.stderr)
