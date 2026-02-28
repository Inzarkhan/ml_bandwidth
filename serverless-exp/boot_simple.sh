workloads=(cpuintensive memory_touch mixed_touch_cpu)
mems=(128 256 512 1024)

for w in "${workloads[@]}"; do
  for m in "${mems[@]}"; do
    for rep in 1 2 3 4 5; do
      docker run --rm \
        --user "$(id -u)":"$(id -g)" \
        --memory "${m}m" --cpus 1 \
        -v "$(pwd)/serverless-exp/logs:/logs" \
        -v "$(pwd)/serverless-exp/workloads:/app/workloads" \
        serverless-runner:ubuntu22 \
        --workload "$w" --mem_limit_mb "$m" --runs 25 --cold_every 5
    done
  done
done