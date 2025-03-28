def get_slo_ms(file_path, percentiles=[50, 90, 99]):
    ttft = dict()
    tbt = dict()
    ete = dict()
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('request_id'):
                continue
            req_id, arrival_time, *finish_time = line.strip().split(',')
            finish_time = finish_time[:-1]
            if 'None' in finish_time:
                continue
            req_ttft = float(finish_time[0]) - float(arrival_time)
            req_ete = float(finish_time[-1]) - float(arrival_time)
            req_tbt = (float(finish_time[-1]) - float(finish_time[0])) / (len(finish_time) - 1)
            ttft[int(req_id)] = req_ttft * 1000
            tbt[int(req_id)] = req_tbt * 1000
            ete[int(req_id)] = req_ete * 1000
    ttft_sorted = sorted(ttft.values())
    tbt_sorted = sorted(tbt.values())
    ete_sorted = sorted(ete.values())
    ttft_p = dict()
    tbt_p = dict()
    ete_p = dict()
    for p in percentiles:
        ttft_p[p] = ttft_sorted[int(len(ttft_sorted) * p / 100)]
        tbt_p[p] = tbt_sorted[int(len(tbt_sorted) * p / 100)]
        ete_p[p] = ete_sorted[int(len(ete_sorted) * p / 100)]
    return ttft_p, tbt_p, ete_p