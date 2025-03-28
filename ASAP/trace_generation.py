# %%
import os
import numpy as np
import pandas as pd

from collections import namedtuple

import requests
from scipy import stats


Distributions = namedtuple('Distributions', ['application_id',
                                             'request_type',
                                             'arrival_process',
                                             'batch_size',
                                             'prompt_size',
                                             'token_size'])
Distribution = namedtuple('Distribution', ['name', 'params'])


def generate_samples(distribution, params, size):
    """
    Generate random samples from the given distribution.
    """
    if distribution == "constant":
        return np.ones(size) * params["value"]
    elif distribution == "normal":
        return stats.norm(**params).rvs(size=size)
    elif distribution == "truncnorm":
        return stats.truncnorm(**params).rvs(size=size)
    elif distribution == "randint":
        return stats.uniform(**params).rvs(size=size)
    elif distribution == "uniform":
        return stats.uniform(**params).rvs(size=size)
    elif distribution == "exponential":
        return stats.expon(**params).rvs(size=size)
    elif distribution == "poisson":
        return stats.poisson(**params).rvs(size=size)
    elif distribution == "trace":
        df = pd.read_csv(params["filename"])
        return df[params["column"]].sample(size, replace=True).values
    else:
        raise ValueError(f"Invalid distribution: {distribution}")


def generate_trace(max_requests, distributions, end_time=None):
    """
    Generate a trace of requests based on the given distributions.
    """
    # Generate request IDs
    request_ids = np.arange(max_requests)

    # Generate the distributions
    arrival_timestamps = generate_samples(distributions.arrival_process.name,
                                          distributions.arrival_process.params,
                                          max_requests)
    arrival_timestamps = np.cumsum(arrival_timestamps)
    application_ids = generate_samples(distributions.application_id.name,
                                       distributions.application_id.params,
                                       max_requests)
    application_ids = map(int, application_ids)
    batch_sizes = generate_samples(distributions.batch_size.name,
                                   distributions.batch_size.params,
                                   max_requests)
    batch_sizes = map(int, batch_sizes)
    prompt_sizes = generate_samples(distributions.prompt_size.name,
                                    distributions.prompt_size.params,
                                    max_requests)
    prompt_sizes = map(int, prompt_sizes)
    token_sizes = generate_samples(distributions.token_size.name,
                                   distributions.token_size.params,
                                   max_requests)
    token_sizes = map(int, token_sizes)
    request_type_ids = generate_samples(distributions.request_type.name,
                                        distributions.request_type.params,
                                        max_requests)
    request_type_ids = map(int, request_type_ids)

    # Combine the arrays into a DataFrame
    trace_df = pd.DataFrame({
        "request_id": request_ids,
        "request_type": request_type_ids,
        "application_id": application_ids,
        "arrival_timestamp": arrival_timestamps,
        "batch_size": batch_sizes,
        "prompt_size": prompt_sizes,
        "token_size": token_sizes,
    })

    if end_time is not None:
        trace_df = trace_df[trace_df["arrival_timestamp"] < end_time]

    return trace_df


def get_exponential_scale(num_servers, utilization, request_duration):
    """
    assumes that request_duration is in seconds
    """
    interarrival_time = request_duration / (1.0 * utilization)
    exponential_scale = interarrival_time / num_servers
    return exponential_scale


def generate_trace_from_utilization(
    max_requests,
    end_time,
    num_servers,
    utilization,
    request_duration,
    pt_distributions_file):
    """
    Generate request traces for the simulator using prompt and token
    size distributions.
    """
    exponential_scale = get_exponential_scale(num_servers, utilization, request_duration)
    distributions = Distributions(
        application_id=Distribution("constant", {"value": 0}),
        request_type=Distribution("constant", {"value": 2}), # 2 is for LLM inference
        arrival_process=Distribution("exponential", {"scale": exponential_scale}),
        prompt_size=Distribution("trace", {"filename": pt_distributions_file,
                                           "column": "ContextTokens"}),
        token_size=Distribution("trace", {"filename": pt_distributions_file,
                                          "column": "GeneratedTokens"}),
        batch_size=Distribution("constant", {"value": 1}),
    )

    trace_df = generate_trace(max_requests,
                              distributions,
                              end_time=end_time)
    return trace_df


def generate_trace_from_prompt_token_size_distributions(
    max_requests,
    end_time,
    request_rate,
    pt_distributions_filename):
    """
    Generate request traces for the simulator using prompt and token
    size distributions.
    """
    distributions = Distributions(
        application_id=Distribution("constant", {"value": 0}),
        request_type=Distribution("constant", {"value": 2}), # 2 is for LLM inference
        arrival_process=Distribution("exponential", {"scale": 1.0 / request_rate}),
        prompt_size=Distribution("trace", {"filename": pt_distributions_filename,
                                           "column": "ContextTokens"}),
        #prompt_size=Distribution("truncnorm", {"a": (prompt_min-prompt_mean)/prompt_std,
        #                                       "b": (prompt_max-prompt_mean)/prompt_std,
        #                                       "loc": prompt_mean,
        #                                       "scale": prompt_std}),
        token_size=Distribution("trace", {"filename": pt_distributions_filename,
                                          "column": "GeneratedTokens"}),
        #token_size=Distribution("truncnorm", {"a": (token_min-token_mean)/token_std,
        #                                      "b": (token_max-token_mean)/token_std,
        #                                      "loc": token_mean,
        #                                      "scale": token_std}),
        batch_size=Distribution("constant", {"value": 1}),
    )
    trace_df = generate_trace(max_requests,
                              distributions,
                              end_time=end_time)
    return trace_df


def generate_traces(max_requests,
                    end_time,
                    request_rates,
                    pt_distributions_file,
                    trace_filename_template):
    """
    Generate traces with prompt/token size distributions.
    """
    for request_rate in request_rates:
        trace_df = generate_trace_from_prompt_token_size_distributions(
            max_requests,
            end_time,
            request_rate,
            pt_distributions_file)
        trace_filename = trace_filename_template.format(request_rate)
        trace_df.to_csv(trace_filename, index=False)


def generate_code_traces(
    max_requests,
    end_time,
    request_rates,
    code_distributions_file,
    trace_filename_template="traces/rr_code_{}.csv"):
    """
    code traces distribution
    prompt_mean = 2048, prompt_std = 1973, prompt_min = 3, prompt_max = 7437
    token_mean = 28, token_std = 60, token_min = 6, token_max = 1899
    """
    if not os.path.exists(trace_filename_template[:trace_filename_template.rfind("/")]):
        os.makedirs(trace_filename_template[:trace_filename_template.rfind("/")])

    generate_traces(max_requests,
                    end_time,
                    request_rates,
                    code_distributions_file,
                    trace_filename_template)


def generate_conv_traces(
    max_requests,
    end_time,
    request_rates,
    conv_distributions_file,
    trace_filename_template="traces/rr_conv_{}.csv"):
    """
    conv traces distribution
    prompt_mean = 1155, prompt_std = 1109, prompt_min = 2, prompt_max = 14050
    token_mean = 211, token_std = 163, token_min = 7, token_max = 1000
    """
    if not os.path.exists(trace_filename_template[:trace_filename_template.rfind("/")]):
        os.makedirs(trace_filename_template[:trace_filename_template.rfind("/")])

    generate_traces(max_requests,
                    end_time,
                    request_rates,
                    conv_distributions_file,
                    trace_filename_template)


def download_file(url, filename):
    """
    Download a file from the given URL.
    """
    response = requests.get(url)
    with open(filename, "wb") as f:
        f.write(response.content)


def download_azure_llm_traces():
    """
    Download traces from the given URL.
    """
    if not os.path.exists("data"):
        os.makedirs("data")

    url_base = "https://raw.githubusercontent.com/Azure/AzurePublicDataset/master/data/"

    if not os.path.exists("data/code_distributions.csv"):
        url = url_base + "AzureLLMInferenceTrace_code.csv"
        download_file(url, "data/code_distributions.csv")
        print("Downloaded code traces")

    if not os.path.exists("data/conv_distributions.csv"):
        url = url_base + "AzureLLMInferenceTrace_conv.csv"
        download_file(url, "data/conv_distributions.csv")
        print("Downloaded conv traces")

# %%
# Generate traces
random_seed = 0

# requests_rates_8k = np.arange(1.0, 2.0, 1.0)
# requests_rates_8k = np.round(np.arange(0.1, 1.0, 0.1), decimals=1)
requests_rates_8k = [0.5, 1, 2, 3, 4, 5, 7, 9, 11, 13, 15, 17, 19]

np.random.seed(random_seed)
overwrite = False
# generate 8k traces if not exists
for req_rate in requests_rates_8k:
    trace_file = f'traces/rr_code_{req_rate}.csv'
    if not os.path.exists(trace_file) or overwrite:
        generate_code_traces(
            max_requests=50000,
            end_time=500,
            request_rates=[req_rate],
            code_distributions_file="data/code_distributions.csv")
    trace_file = f'traces/rr_conv_{req_rate}.csv'
    if not os.path.exists(trace_file) or overwrite:
        generate_conv_traces(
            max_requests=50000,
            end_time=500,
            request_rates=[req_rate],
            conv_distributions_file="data/conv_distributions.csv")

print("Generated 8k traces at different request rates")

# Generate long context traces with the same ratio
# for max_len in [32, 128]:
#     times = max_len // 8
#     if max_len == 32:
#         long_ctx_requests_rates = requests_rates_32k
#     elif max_len == 128:
#         long_ctx_requests_rates = requests_rates_128k
#     for req_rate in long_ctx_requests_rates:
#         for workload in ['code', 'conv']:
#             trace_file = f'traces/rr_{workload}_{req_rate}.csv'
#             long_trace_file = f'traces/rr_{workload}{max_len}k_{req_rate}.csv'
#             if not os.path.exists(long_trace_file) or overwrite:
#                 with open(trace_file, 'r') as f:
#                     with open(long_trace_file, 'w') as f2:
#                         for line in f:
#                             if line.startswith('request_id'):
#                                 f2.write(line)
#                                 continue
#                             req_id, req_type, app_id, arrival_time, batch_size, prompt_size, token_size = line.strip().split(',')
#                             f2.write(f'{req_id},{req_type},{app_id},{arrival_time},{batch_size},{int(prompt_size)*times},{int(token_size)*times}\n')
# print("Generated long context traces at different request rates")
# %%
np.random.seed(1)
generate_code_traces(
        max_requests=50000,
        end_time=500,
        request_rates=[15],
        code_distributions_file="data/code_distributions.csv")
# %%
# Generate long ctx distribution files
ctx_exp = {
            '32k': (13, 15), 
           '128k': (15, 17)}
ctx_ratio_exp = {
                 '32k': [4, 7], 
                 '128k': [4, 7]}
all_ctx = dict()
all_ratio = dict()

# ctx len
for ctx_len in ['32k', '128k']:
    mu = ctx_exp[ctx_len][0]
    maxx = ctx_exp[ctx_len][1]
    sigma = 1
    s = np.random.normal(mu, sigma, 10000)
    np.clip(s, 0, maxx, out=s)
    print(f'Num of {ctx_len}: {len([x for x in s if x == maxx])}')
    ctxx = 2**s
    ctxx = ctxx.astype(int)
    all_ctx[ctx_len] = ctxx
# ratio
all_ratio_exps = [4, 7]
for ratio_exp in all_ratio_exps:
    mu = ratio_exp
    sigma = 1
    s = np.random.normal(mu, sigma, 10000)
    ratioo = 2**s
    all_ratio[ratio_exp] = ratioo

for ctx_len in ['32k', '128k']:
    for exp in ctx_ratio_exp[ctx_len]:
        ratio = 2**exp
        distribution_file = f'data/synthetic_{ctx_len}_{ratio}.csv'
        if not os.path.exists(distribution_file):
            with open(distribution_file, 'w') as f:
                f.write('TIMESTAMP,ContextTokens,GeneratedTokens\n')
                for i in range(10000):
                    total_ctx = all_ctx[ctx_len][i]
                    trace_ratio = all_ratio[exp][i]
                    ctx_tokens = int(total_ctx * trace_ratio / (trace_ratio + 1))
                    gen_tokens = total_ctx - ctx_tokens
                    f.write(f'{i},{ctx_tokens},{gen_tokens}\n')
            print(f'Generated {distribution_file}')

requests_rates_32k = np.round(np.arange(0.5, 2.0, 0.2), decimals=1)
requests_rates_128k = np.round(np.arange(0.1, 1.0, 0.1),decimals=1)

for ctx_len in ['32k', '128k']:
    if ctx_len == '32k':
        requests_rates = requests_rates_32k
    elif ctx_len == '128k':
        requests_rates = requests_rates_128k
    for exp in ctx_ratio_exp[ctx_len]:
        ratio = 2**exp
        for req_rate in requests_rates:
            trace_file = f'traces/rr_{ctx_len}_{ratio}_{req_rate}.csv'
            if not os.path.exists(trace_file) or overwrite:
                trace_filename_template = f'traces/rr_{ctx_len}_{ratio}_{{}}.csv'
                if not os.path.exists(trace_filename_template[:trace_filename_template.rfind("/")]):
                    os.makedirs(trace_filename_template[:trace_filename_template.rfind("/")])

                generate_traces(max_requests=50000,
                                end_time=500,
                                request_rates=[req_rate],
                                pt_distributions_file = f'data/synthetic_{ctx_len}_{ratio}.csv',
                                trace_filename_template=trace_filename_template)

# %%
