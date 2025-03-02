pub fn num_cpu_threads() -> usize {
    std::thread::available_parallelism().unwrap().get()
}