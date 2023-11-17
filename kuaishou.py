#!/usr/bin/env python
"""
Simple programs that gets all BLAS GEMMs from an executable and tries to tune them.
"""
import argparse
import subprocess
import os
import re
import csv
from multiprocessing import Process, Queue
# TODO: Need to figure out this relative path
from rocBlasFinder import rocBlasFinder


class GEMM:
    """
    Class to contain a gemm and its occurances.
    """

    STRIDED_BATCHED_ROCBLAS_BENCH_RE = (
        r"./rocblas-bench -f gemm_strided_batched_ex"
        r" --transposeA (?P<TRANSPOSE_A>\w)"
        r" --transposeB (?P<TRANSPOSE_B>\w)"
        r" -m (?P<M>\d+)"
        r" -n (?P<N>\d+)"
        r" -k (?P<K>\d+)"
        r" --alpha (?P<ALPHA>\d+)"
        r" --a_type (?P<A_TYPE>\w+)"
        r" --lda (?P<LDA>\d+)"
        r" --stride_a (?P<STRIDE_A>\d+)"
        r" --b_type (?P<B_TYPE>\w+)"
        r" --ldb (?P<LDB>\d+)"
        r" --stride_b (?P<STRIDE_B>\d+)"
        r" --beta (?P<BETA>\d+)"
        r" --c_type (?P<C_TYPE>\w+)"
        r" --ldc (?P<LDC>\d+)"
        r" --stride_c (?P<STRIDE_C>\d+)"
        r" --d_type (?P<D_TYPE>\w+)"
        r" --ldd (?P<LDD>\d+)"
        r" --stride_d (?P<STRIDE_D>\d+)"
        r" --batch_count (?P<BATCH_COUNT>\d+)"
        r" --compute_type (?P<COMPUTE_TYPE>\w+)"
        r" --algo (?P<ALGO>\d+)"
        r" --solution_index (?P<SOLUTION_INDEX>\d+)"
        r" --flags (?P<FLAGS>\w+)"
    )

    GENERIC_ROCBLAS_BENCH_RE = (
        r"./rocblas-bench -f gemm_ex"
        r" --transposeA (?P<TRANSPOSE_A>\w)"
        r" --transposeB (?P<TRANSPOSE_B>\w)"
        r" -m (?P<M>\d+)"
        r" -n (?P<N>\d+)"
        r" -k (?P<K>\d+)"
        r" --alpha (?P<ALPHA>\d+)"
        r" --a_type (?P<A_TYPE>\w+)"
        r" --lda (?P<LDA>\d+)"
        r" --b_type (?P<B_TYPE>\w+)"
        r" --ldb (?P<LDB>\d+)"
        r" --beta (?P<BETA>\d+)"
        r" --c_type (?P<C_TYPE>\w+)"
        r" --ldc (?P<LDC>\d+)"
        r" --d_type (?P<D_TYPE>\w+)"
        r" --ldd (?P<LDD>\d+)"
        r" --compute_type (?P<COMPUTE_TYPE>\w+)"
        r" --algo (?P<ALGO>\d+)"
        r" --solution_index (?P<SOLUTION_INDEX>\d+)"
        r" --flags (?P<FLAGS>\w+)"
    )

    def __init__(self, rocblas_bench_string):
        # First match the gemm
        self.rocblas_bench_string = rocblas_bench_string
        if match := re.match(self.GENERIC_ROCBLAS_BENCH_RE, rocblas_bench_string):
            self.match = True
            self.gemm_type = "Generic"
        elif match := re.match(
            self.STRIDED_BATCHED_ROCBLAS_BENCH_RE, rocblas_bench_string
        ):
            self.match = True
            self.gemm_type = "Strided batched"
        else:
            self.match = False

        # Collect data in new if so we can share code
        if self.match:
            self.count = 1
            self.tA = match.group("TRANSPOSE_A")
            self.tB = match.group("TRANSPOSE_B")
            self.m = int(match.group("M"))
            self.n = int(match.group("N"))
            self.k = int(match.group("K"))
            self.alpha = float(match.group("ALPHA"))
            self.lda = int(match.group("LDA"))
            self.ldb = int(match.group("LDB"))
            self.beta = float(match.group("BETA"))
            self.ldc = int(match.group("LDC"))
            self.compute_type = match.group("COMPUTE_TYPE")
            self.a_type = match.group("A_TYPE")
            self.solution_index = match.group("SOLUTION_INDEX")
            if self.gemm_type == "Generic":
                self.key = f"ta:{self.tA},tb:{self.tB},m:{self.m},n{self.n},k{self.k}"
            elif self.gemm_type == "Strided batched":
                self.stride_a = int(match.group("STRIDE_A"))
                self.stride_b = int(match.group("STRIDE_B"))
                self.stride_c = int(match.group("STRIDE_C"))
                self.stride_d = int(match.group("STRIDE_D"))
                self.batch_count = int(match.group("BATCH_COUNT"))
                self.key = f"ta:{self.tA},tb:{self.tB},m:{self.m},n{self.n},k{self.k},sa:{self.stride_a},sb:{self.stride_b},sc:{self.stride_c},bc:{self.batch_count}"

    def __bool__(self):
        return self.match

    def inc_count(self, number=1):
        self.count += number

    def __repr__(self):
        return f"Instances: {self.count} M: {self.m} N: {self.n} K: {self.k} solution_index: {self.solution_index}\n"

    def run_args(self):
        if self.gemm_type == "Generic":
            return self.tA, self.tB, self.m, self.n, self.k, self.alpha, self.beta, self.a_type, self.a_type
        elif self.gemm_type == "Strided batched":
            return (
                self.tA,
                self.tB,
                self.m,
                self.n,
                self.k,
                self.alpha,
                self.beta,
                self.stride_a,
                self.stride_b,
                self.stride_c,
                self.batch_count,
                self.a_type,
                self.a_type
            )

    def csv_list(self):
        # Only two possible formats? from snooping: UserDrivenTuningParser.cpp in tensile
        if self.gemm_type == "Generic":
            return [
                self.tA,
                self.tB,
                self.m,
                self.n,
                1,
                self.k,
                self.alpha,
                self.beta,
                self.lda,
                self.ldb,
                self.ldc,
                self.a_type,
                self.a_type,
                self.compute_type,
                self.solution_index,
            ]
        else:
            return [
                self.tA,
                self.tB,
                self.m,
                self.n,
                self.batch_count,
                self.k,
                self.alpha,
                self.beta,
                self.lda,
                self.ldb,
                self.ldc,
                self.stride_a,
                self.stride_b,
                self.stride_c,
                self.a_type,
                self.a_type,
                self.compute_type,
                self.solution_index,
            ]


class ExecutableRunner:
    """
    Class for running any executable, with the correct env variable, and collect logs
    """

    def __init__(self, executable):
        self.executable = executable

    def run_and_collect(self, show_output=False):
        env = os.environ.copy()
        env["ROCBLAS_LAYER"] = "2"
        # TODO: Needs to swap to "4" and read csv
        process = subprocess.run(
            self.executable,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        self.process_output = process.stdout
        if show_output:
            print(f"Output from subprocess.run: {self.process_output}")

    def get_unique_gemms(self):
        """
        Return every unique gemm with the form [Count, TransposeA, TransposeB, M, N, K]
        """
        out_dict = {}
        lines = self.process_output.splitlines()
        for line in lines:
            if gemm := GEMM(line):
                # TODO Seems like there should be a better way?
                if gemm.key in out_dict:
                    out_dict[gemm.key].inc_count()
                else:
                    out_dict[gemm.key] = gemm
        return list(out_dict.values())

def run_tuning(gpu_id, in_q, out_q):
    tunner = rocBlasFinder(gpu_id)
    while not in_q.empty():
        gemm = in_q.get()
        results = tunner.run(*gemm.run_args())
        # TODO: Check if bad?
        match = re.match(
            r"Default: (\d+.\d+) Winner: (\d+.\d+) Solution: (\d+)", results
        )
        default_time = float(match.group(1))
        winning_time = float(match.group(2))
        solution_nu = int(match.group(3))
        old_time = int(gemm.count) * default_time
        new_time = int(gemm.count) * winning_time
        # Write new solution to gemm
        gemm.solution_index = solution_nu
        if new_time<old_time:
            out_q.put((gemm, old_time, new_time))

def process_gemms(gemms):
    gpu_ids = [int(gpu_id) for gpu_id in os.environ.get('HIP_VISIBLE_DEVICES', '0').split(',')]
    in_q = Queue()
    out_q = Queue()

    for gemm in gemms:
        in_q.put(gemm)

    processes = []
    for gpu_id in range(len(gpu_ids)):
        p = Process(target=run_tuning, args=(gpu_id, in_q, out_q))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
        p.close()

    total_old = 0
    total_new = 0
    gemms = []
    while not out_q.empty():
        gemm, old_time, new_time = out_q.get()
        gemms.append(gemm)
        total_old += old_time
        total_new += new_time
    return gemms, total_old, total_new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-o",
        help="Output file with the results. NOT IMPLEMENTED YET",
        action="store",
        dest="output",
        default="BlasterOutput.csv",
    )
    parser.add_argument("--show_gemms", action="store_true")
    args = parser.parse_args()

    # Run and collect
    # executable = ExecutableRunner(args.executable)
    # executable.run_and_collect()
    print(f"{os.linesep}{'>'*20:<20}{' rocBlas Output ':^20}{'<'*20:>20}{os.linesep}")
    # gemms = executable.get_unique_gemms()
    with open('./optimize.csv', 'w') as f:
        f.write('transA,transB,M,N,K,solutions,Default time(ns),Winner time(ns),best sol,Improved')
    # gemms_args = [(2048, 3072, 8192), (2048, 1024, 8192), (2048, 5504, 8192), (2048, 8192, 2752)]
    gemms_args = [(15000,1,32), (15000,1,128), (15000,3,1024), (15000,32,64), (15000,8,32), (15000,64,128), (1, 256, 3), (15000,512,1024)]
    # for i in range(1, 21):
    # for i in range(1, 2):
    #     gemms_args.append((i, 8192, 1024))
    #     gemms_args.append((i, 2752, 8192))
    gemms = []
    # for m, n, k in gemms_args:
        # gemms.append(GEMM(f"./rocblas-bench -f gemm_ex --transposeA N --transposeB N -m {m} -n {n} -k {k} --alpha 1 --a_type bf16_r --lda 4096 --b_type bf16_r --ldb 4096 --beta 0 --c_type bf16_r --ldc 4096 --d_type bf16_r --ldd 4096 --compute_type f32_r --algo 0 --solution_index 0 --flags 0"))

    gemms.append(GEMM('./rocblas-bench -f gemm_ex --transposeA N --transposeB N -m 1 -n 15000 -k 32 --alpha 1 --a_type bf16_r --lda 1 --b_type bf16_r --ldb 32 --beta 0 --c_type bf16_r --ldc 1 --d_type bf16_r --ldd 1 --compute_type f32_r --algo 0 --solution_index 0 --flags 0'))
    gemms.append(GEMM('./rocblas-bench -f gemm_ex --transposeA N --transposeB N -m 1 -n 15000 -k 128 --alpha 1 --a_type bf16_r --lda 1 --b_type bf16_r --ldb 128 --beta 0 --c_type bf16_r --ldc 1 --d_type bf16_r --ldd 1 --compute_type f32_r --algo 0 --solution_index 0 --flags 0'))
    gemms.append(GEMM('./rocblas-bench -f gemm_ex --transposeA N --transposeB N -m 3 -n 15000 -k 1024 --alpha 1 --a_type bf16_r --lda 3 --b_type bf16_r --ldb 1024 --beta 0 --c_type bf16_r --ldc 3 --d_type bf16_r --ldd 3 --compute_type f32_r --algo 0 --solution_index 0 --flags 0'))
    gemms.append(GEMM('./rocblas-bench -f gemm_ex --transposeA N --transposeB N -m 32 -n 15000 -k 64 --alpha 1 --a_type bf16_r --lda 32 --b_type bf16_r --ldb 64 --beta 0 --c_type bf16_r --ldc 32 --d_type bf16_r --ldd 32 --compute_type f32_r --algo 0 --solution_index 0 --flags 0'))
    gemms.append(GEMM('./rocblas-bench -f gemm_ex --transposeA N --transposeB N -m 8 -n 15000 -k 32 --alpha 1 --a_type bf16_r --lda 8 --b_type bf16_r --ldb 32 --beta 0 --c_type bf16_r --ldc 8 --d_type bf16_r --ldd 8 --compute_type f32_r --algo 0 --solution_index 0 --flags 0'))
    gemms.append(GEMM('./rocblas-bench -f gemm_ex --transposeA N --transposeB N -m 64 -n 15000 -k 128 --alpha 1 --a_type bf16_r --lda 64 --b_type bf16_r --ldb 128 --beta 0 --c_type bf16_r --ldc 64 --d_type bf16_r --ldd 64 --compute_type f32_r --algo 0 --solution_index 0 --flags 0'))
    gemms.append(GEMM('./rocblas-bench -f gemm_strided_batched_ex --transposeA N --transposeB N -m 256 -n 1 -k 3 --alpha 1 --a_type bf16_r --lda 256 --stride_a 768 --b_type bf16_r --ldb 3 --stride_b 3 --beta 0 --c_type bf16_r --ldc 256 --stride_c 256 --d_type bf16_r --ldd 256 --stride_d 256 --batch_count 15000 --compute_type f32_r --algo 0 --solution_index 0 --flags 0'))
    gemms.append(GEMM('./rocblas-bench -f gemm_ex --transposeA N --transposeB N -m 512 -n 15000 -k 1024 --alpha 1 --a_type bf16_r --lda 512 --b_type bf16_r --ldb 1024 --beta 0 --c_type bf16_r --ldc 512 --d_type bf16_r --ldd 512 --compute_type f32_r --algo 0 --solution_index 0 --flags 0'))
        # gemms.append(GEMM(f"./rocblas-bench -f gemm_strided_batched_ex --transposeA N --transposeB N -m {m} -n {n} -k {k} --alpha 1 --a_type bf16_r --lda 4096 --b_type bf16_r --ldb 4096 --beta 0 --c_type bf16_r --ldc 4096 --d_type bf16_r --ldd 4096 --compute_type f32_r --algo 0 --solution_index 0 --flags 0"))
    if args.show_gemms:
        print(f"Got unique gemms {gemms}")

    gemms, total_old, total_new = process_gemms(gemms)
    
    print(
        f"{os.linesep}{'>'*20:<20}{' Summary ':^20}{'<'*20:>20}{os.linesep}"
        f"Old time: {total_old}{os.linesep}"
        f"New time: {total_new}{os.linesep}"
        f"Total improvement: {(total_old-total_new)/total_old:0.2f}"
    )

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        headers = [
            "transA",
            "transB",
            "M",
            "N",
            "batch_count",
            "K",
            "alpha",
            "beta",
            "lda",
            "ldb",
            "ldc",
            "input_type",
            "output_type",
            "compute_type",
            "solution_index",
        ]
        writer.writerow(headers)
        for gemm in gemms:
            writer.writerow(gemm.csv_list())


if __name__ == "__main__":
    main()

