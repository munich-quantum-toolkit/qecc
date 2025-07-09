# Copyright (c) 2023 - 2025 Chair for Design Automation, TUM
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Licensed under the MIT License

"""This module contains utility functions for loading and processing raw simulation data."""

from __future__ import annotations

import itertools
import json
import os
from dataclasses import dataclass, fields
from json.decoder import JSONDecodeError
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class BpParams:
    """Class to store parameters for BP decoding."""

    bp_method: str = "msl"
    max_bp_iter: int = 30
    osd_order: int = 10
    osd_method: str = "osd_cs"
    ms_scaling_factor: float = 0.75
    schedule: str = "parallel"
    omp_thread_count: int = 1
    random_serial_schedule: int = 0
    serial_schedule_order: list[int] | None = None
    cutoff: float = np.inf

    @classmethod
    def from_dict(cls, dict_: dict[str, Any]) -> BpParams:
        """Creates a BpParams object from a dictionary."""
        class_fields = {f.name for f in fields(cls)}
        return BpParams(**{k: v for k, v in dict_.items() if k in class_fields})


def extract_settings(filename: str) -> dict[str, list[str]]:
    """Extracts all settings from a parameter file and returns them as a dictionary."""
    keyword_lists: dict[str, list[str]] = {}

    with Path(filename).open(encoding="utf-8") as file:
        for line in file:
            json_data = json.loads(line.strip())
            for keyword, value in json_data.items():
                if keyword not in keyword_lists:
                    keyword_lists[keyword] = []
                if value not in keyword_lists[keyword]:
                    keyword_lists[keyword].append(value)

    return keyword_lists


def load_data(
    input_filenames: list[str],
) -> list[dict[str, str]]:
    """Loads data from a list of JSON files and returns it as a list of dictionaries."""
    data = []
    for file in input_filenames:
        path = Path(file)

        try:
            ldata = json.load(path.open())
            data.append(ldata)
        except json.decoder.JSONDecodeError:
            merge_json_files(str(path.with_suffix("")))
            ldata = json.load(path.open())
            data.append(ldata)
    return data


def calculate_error_rates(
    success_cnt: int, runs: int, code_params: dict[str, int]
) -> tuple[float, float, float, float]:
    """Calculates logical error rates."""
    logical_err_rate = 1.0 - (success_cnt / runs)
    logical_err_rate_eb = np.sqrt((1 - logical_err_rate) * logical_err_rate / runs)
    word_error_rate = 1.0 - (1 - logical_err_rate) ** (1 / code_params["k"])
    word_error_rate_eb = (
        logical_err_rate_eb * ((1 - logical_err_rate_eb) ** (1 / code_params["k"] - 1)) / code_params["k"]
    )
    return (
        logical_err_rate,
        logical_err_rate_eb,
        word_error_rate,
        word_error_rate_eb,
    )


def is_converged(x_success: int, z_success: int, runs: int, code_params: dict[str, int], precision: float) -> bool:
    """Checks if the logical error rates for X and Z are converged."""
    x_cond = _check_convergence(x_success, runs, code_params, precision_cutoff=precision)
    z_cond = _check_convergence(z_success, runs, code_params, precision_cutoff=precision)
    return x_cond == z_cond is True


def _check_convergence(
    success_cnt: int, runs: int, code_params: dict[str, int], precision_cutoff: float
) -> bool | None:
    _, _, ler, ler_eb = calculate_error_rates(success_cnt, runs, code_params)
    if ler_eb != 0.0:
        if ler_eb / ler < precision_cutoff:
            return True
        return None
    return False


def create_outpath(
    x_meta: bool = False,
    z_meta: bool = False,
    bias: list[float] | None = None,
    codename: str | None = None,
    single_stage: bool = True,
    sus_th_depth: int | None = None,
    rounds: int | None = None,
    identifier: int = 0,
    analog_info: bool = False,
    analog_tg: bool = False,
    repetitions: int | None = None,
    experiment: str = "wer_per_round",
    **kwargs: dict[str, Any],
) -> str:
    """Creates a path for storing simulation results."""
    path = f"results/{experiment:s}/"
    if analog_info:
        path += "analog_info/"  # ranvendraan et al analog info decoder
    elif analog_tg:
        path += "analog_tg/"  # analog tannergraph bp
    else:
        path += "hard_syndrome/"  # standard hard syndrome decoding only
    if bias is not None:
        path += f"single_stage={single_stage}/bias={bias[0]}_{bias[1]}_{bias[2]}/"

    if sus_th_depth:
        path += f"sus_th_depth={sus_th_depth}/"
    elif rounds:
        path += f"rounds={rounds}/"

    if repetitions:
        path += f"repetitions={repetitions}/"

    if x_meta:
        path += "x-meta=true/"
    else:
        path += "x-meta=false/"

    if z_meta:
        path += "z-meta=true/"
    else:
        path += "z-meta=false/"

    path += f"{codename:s}/"

    if "syndr_err_rate" not in kwargs or kwargs["syndr_err_rate"] is None:
        if "sigma" in kwargs:
            path += f"per_{kwargs['data_err_rate']:.3e}_sigma_{kwargs['sigma']:.3e}/"
        if "z_sigma" in kwargs:
            path += f"per_{kwargs['data_err_rate']:.3e}_x_sigma_{kwargs['x_sigma']:.3e}_z_sigma_{kwargs['z_sigma']:.3e}"
    else:
        path += f"per_{kwargs['data_err_rate']:.3e}_ser_{kwargs['syndr_err_rate']:.3e}/"

    Path(path).mkdir(parents=True, exist_ok=True)

    f_loc = path + f"id_{identifier}.json"
    while Path(f_loc).exists():
        identifier += 1
        f_loc = path + f"id_{identifier}.json"

    Path(f_loc).touch()
    return f_loc


def replace_inf(lst: list[str]) -> list[str]:
    """Replaces all occurrences of np.inf in a list with the string "i"."""
    new_lst = []
    for item in lst:
        if np.isinf(item):
            new_lst.append("i")
        else:
            new_lst.append(item)
    return new_lst


def product_dict(**kwargs: Any) -> Any:  # noqa: ANN401
    """Generate a iterator of dictionaries where each dictionary is a cartesian product.

    of the values associated with each key in the input dictionary.
    """
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        yield dict(zip(keys, instance))


def zip_dict(**kwargs: dict[str, Any]) -> Any:  # noqa: ANN401
    """Create a iterator of dictionaries where each dictionary contains the zip() of the values associated with each key in the input dictionary."""
    return (dict(zip(kwargs.keys(), values)) for values in zip(*kwargs.values()))


def _update_error_rates(success_cnt: int, runs: int, code_k: int) -> tuple[float, float, float, float]:
    """Calculates logical error rate, logical error rate error bar, word error rate, and word error rate error bar."""
    logical_err_rate = 1.0 - (success_cnt / runs)
    logical_err_rate_eb = np.sqrt((1 - logical_err_rate) * logical_err_rate / runs)
    word_error_rate = 1.0 - (1 - logical_err_rate) ** (1 / code_k)
    word_error_rate_eb = logical_err_rate_eb * ((1 - logical_err_rate_eb) ** (1 / code_k - 1)) / code_k
    return (
        logical_err_rate,
        logical_err_rate_eb,
        word_error_rate,
        word_error_rate_eb,
    )


def merge_datasets(datasets: list[dict[str, Any]]) -> dict[str, Any]:
    """Merges a list of dictionaries into a single dictionary.

    The values for the fields "nr_runs", "x_success_cnt" and "z_success_cnt" are extracted from each dictionary and added together.

    Args:
        datasets (List[Dict[str, Any]]): A list of dictionaries to be merged.

    Returns:
        Dict[str, Any]: A dictionary containing the merged data.
    """
    if not datasets:
        return {}

    # Start with a copy of the first dictionary in the list
    merged_data = dict(datasets[0])

    # Extract and add up the values for "nr_runs", "x_success_cnt", and "z_success_cnt"
    for data in datasets[1:]:
        merged_data["nr_runs"] += data.get("nr_runs", 0)
        merged_data["x_success_cnt"] += data.get("x_success_cnt", 0)
        merged_data["z_success_cnt"] += data.get("z_success_cnt", 0)

    # Update logical and word error rates based on accumulated data.
    runs = merged_data["nr_runs"]
    x_success_cnt = merged_data["x_success_cnt"]
    z_success_cnt = merged_data["z_success_cnt"]

    x_ler, x_ler_eb, x_wer, x_wer_eb = _update_error_rates(x_success_cnt, runs, merged_data["code_K"])
    z_ler, z_ler_eb, z_wer, z_wer_eb = _update_error_rates(z_success_cnt, runs, merged_data["code_K"])

    update_dict = {
        "x_ler": x_ler,
        "x_ler_eb": x_ler_eb,
        "x_wer": x_wer,
        "x_wer_eb": x_wer_eb,
        "x_success_cnt": x_success_cnt,
        "z_ler": z_ler,
        "z_ler_eb": z_ler_eb,
        "z_wer": z_wer,
        "z_wer_eb": z_wer_eb,
        "z_success_cnt": z_success_cnt,
    }
    merged_data.update(update_dict)
    return merged_data


def _merge_datasets_x(datasets_: list[dict[str, Any]]) -> dict[str, Any]:
    """Merges a list of dictionaries into a single dictionary.

    The values for the fields "nr_runs", "x_success_cnt" and "z_success_cnt" are extracted from each dictionary and added together.

    Args:
        datasets_ (List[Dict[str, Any]]): A list of dictionaries to be merged.

    Returns:
        Dict[str, Any]: A dictionary containing the merged data.
    """
    # remove datasets that do not contain x_success_cnt
    datasets = [data for data in datasets_ if "x_success_cnt" in data]

    if not datasets:
        return {}

    # Start with a copy of the first dictionary in the list that contains z_success_cnt
    # and remove that dict from the list
    merged_data = {}
    for i, data in enumerate(datasets):
        if "x_success_cnt" in data:
            merged_data = dict(datasets.pop(i))
            break

    # Extract and add up the values for "nr_runs", "x_success_cnt", and "z_success_cnt"
    for data in datasets:
        merged_data["nr_runs"] += data.get("nr_runs", 0)
        merged_data["x_success_cnt"] += data.get("x_success_cnt", 0)

    # Update logical and word error rates based on accumulated data.
    runs = merged_data["nr_runs"]
    x_success_cnt = merged_data["x_success_cnt"]

    x_ler, x_ler_eb, x_wer, x_wer_eb = _update_error_rates(x_success_cnt, runs, merged_data["code_K"])

    update_dict = {
        "x_ler": x_ler,
        "x_ler_eb": x_ler_eb,
        "x_wer": x_wer,
        "x_wer_eb": x_wer_eb,
        "x_success_cnt": x_success_cnt,
    }
    merged_data.update(update_dict)
    return merged_data


def _merge_datasets_z(datasets_: list[dict[str, Any]]) -> dict[str, Any]:
    """Merges a list of dictionaries into a single dictionary. The values for the fields "nr_runs".

    "x_success_cnt" and "z_success_cnt" are extracted from each dictionary and added together.

    Args:
        datasets_ (List[Dict[str, Any]]): A list of dictionaries to be merged.

    Returns:
        Dict[str, Any]: A dictionary containing the merged data.
    """
    # remove datasets that do not contain z_success_cnt
    datasets = [data for data in datasets_ if "z_success_cnt" in data]

    if not datasets:
        return {}

    # Start with a copy of the first dictionary in the list that contains z_success_cnt
    # and remove that dict from the list
    merged_data = {}
    for i, data in enumerate(datasets):
        if "z_success_cnt" in data:
            merged_data = dict(datasets.pop(i))
            break

    # Extract and add up the values for "nr_runs", "x_success_cnt", and "z_success_cnt"
    for data in datasets:
        merged_data["nr_runs"] += data.get("nr_runs", 0)
        merged_data["z_success_cnt"] += data.get("z_success_cnt", 0)

    # Update logical and word error rates based on accumulated data.
    runs = merged_data["nr_runs"]
    z_success_cnt = merged_data["z_success_cnt"]

    z_ler, z_ler_eb, z_wer, z_wer_eb = _update_error_rates(z_success_cnt, runs, merged_data["code_K"])

    update_dict = {
        "z_ler": z_ler,
        "z_ler_eb": z_ler_eb,
        "z_wer": z_wer,
        "z_wer_eb": z_wer_eb,
        "z_success_cnt": z_success_cnt,
    }
    merged_data.update(update_dict)
    return merged_data


def merge_json_files(input_path: str) -> None:
    """Iterates through all subfolders in the input folder, loads all JSON files in each subfolder.

    and merges them using the `merge_datasets` function. The resulting dictionaries are
    stored in a list and then dumped as a JSON file in the input folder.

    Args:
        input_path (str): The path to the input folder.

    Returns:
        None
    """
    output_data: list[dict[str, Any]] = []
    for folder_name in Path(input_path).iterdir():
        folder_path = input_path / folder_name
        if not folder_path.is_dir():
            continue

        data: list[dict[str, Any]] = []
        for filename in folder_path.iterdir():
            if filename.suffix == ".json":
                file_path = folder_path / filename
                with file_path.open() as file:
                    try:
                        json_data = json.load(file)
                        data.append(json_data)
                    except JSONDecodeError:
                        # don't care about json decode error here
                        pass
        merged_data = merge_datasets(data)
        if merged_data:
            output_data.append(merged_data)

    # save output to parent directory
    code_name = Path(input_path).name
    parent_dir = (Path(input_path) / os.pardir).resolve()
    with (parent_dir / f"{code_name:s}.json").open("w") as output_file:
        json.dump(output_data, output_file, ensure_ascii=False, indent=4)


def merge_json_files_x(input_path: str) -> None:
    """Iterates through all subfolders in the input folder, loads all JSON files in each subfolder.

    and merges them using the `merge_datasets` function. The resulting dictionaries are
    stored in a list and then dumped as a JSON file in the input folder.

    Args:
        input_path (str): The path to the input folder.

    Returns:
        None
    """
    output_data: list[dict[str, Any]] = []
    for folder_name in Path(input_path).iterdir():
        folder_path = input_path / folder_name
        if not folder_path.is_dir():
            continue

        data: list[dict[str, Any]] = []
        for filename in folder_path.iterdir():
            if filename.suffix == ".json":
                with (folder_path / filename).open() as file:
                    try:
                        json_data = json.load(file)
                        data.append(json_data)
                    except JSONDecodeError:
                        # don't care about json decode error here
                        pass
        merged_data = _merge_datasets_x(data)
        if merged_data:
            output_data.append(merged_data)

    # save output to parent directory
    code_name = Path(input_path).name
    parent_dir = (Path(input_path) / os.pardir).resolve()
    with (parent_dir / f"{code_name:s}.json").open("w") as output_file:
        json.dump(output_data, output_file, ensure_ascii=False, indent=4)


def merge_json_files_z(input_path: str) -> None:
    """Iterates through all subfolders in the input folder, loads all JSON files in each subfolder.

    and merges them using the `merge_datasets` function. The resulting dictionaries are
    stored in a list and then dumped as a JSON file in the input folder.

    Args:
        input_path (str): The path to the input folder.

    Returns:
        None
    """
    output_data: list[dict[str, Any]] = []
    for folder_name in Path(input_path).iterdir():
        folder_path = input_path / folder_name

        if not folder_path.is_dir():
            continue

        data: list[dict[str, Any]] = []
        for filename in folder_path.iterdir():
            if filename.suffix == ".json":
                with (folder_path / filename).open() as file:
                    try:
                        json_data = json.load(file)
                        data.append(json_data)
                    except JSONDecodeError:
                        # don't care about json decode error here
                        pass
        merged_data = _merge_datasets_z(data)
        if merged_data:
            output_data.append(merged_data)

    # save output to parent directory
    code_name = Path(input_path).name
    parent_dir = (Path(input_path) / os.pardir).resolve()
    with (parent_dir / f"{code_name:s}.json").open("w") as output_file:
        json.dump(output_data, output_file, ensure_ascii=False, indent=4)


def merge_json_files_xz(input_path: str) -> None:
    """Iterates through all subfolders in the input folder, loads all JSON files in each subfolder.

    and merges them using the `merge_datasets` function. The resulting dictionaries are
    stored in a list and then dumped as a JSON file in the input folder.

    Args:
        input_path (str): The path to the input folder.

    Returns:
        None
    """
    output_data: list[dict[str, Any]] = []
    for folder_name in Path(input_path).iterdir():
        folder_path = input_path / folder_name
        if not folder_path.is_dir():
            continue

        data: list[dict[str, Any]] = []
        for filename in folder_path.iterdir():
            if filename.suffix == ".json":
                with (folder_path / filename).open() as file:
                    try:
                        json_data = json.load(file)
                        data.append(json_data)
                    except JSONDecodeError:
                        # don't care about json decode error here
                        pass
        merged_data_x = _merge_datasets_x(data)
        merged_data_z = _merge_datasets_z(data)
        merged_data = _combine_xz_data(merged_data_x, merged_data_z)
        if merged_data:
            output_data.append(merged_data)

    # save output to parent directory
    code_name = Path(input_path).name
    parent_dir = (Path(input_path) / os.pardir).resolve()
    with (parent_dir / f"{code_name:s}.json").open("w") as output_file:
        json.dump(output_data, output_file, ensure_ascii=False, indent=4)


def _combine_xz_data(xdata: dict[str, Any] | None, zdata: dict[str, Any] | None) -> dict[str, Any]:
    """Combine the x and z data into a single dictionary.

    Before doing that, rename "runs" in each dictionary to "x_runs" and "z_runs" respectively.

    """
    if xdata and zdata:
        # print(xdata)
        xdata["x_runs"] = xdata.pop("nr_runs")
        zdata["z_runs"] = zdata.pop("nr_runs")
        xdata.update(zdata)
        return xdata
    if xdata:
        xdata["x_runs"] = xdata.pop("nr_runs")
        return xdata
    if zdata:
        zdata["z_runs"] = zdata.pop("nr_runs")
        return zdata

    return {}
