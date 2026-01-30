import mlx.core as mx

a = mx.random.uniform(shape=(512, 512))
b = mx.random.uniform(shape=(512, 512))
mx.eval(a, b)

trace_file = "mlx_trace.gputrace"

# Make sure to run with MTL_CAPTURE_ENABLED=1 and
# that the path trace_file does not already exist.
mx.metal.start_capture(trace_file)

for _ in range(10):
  mx.eval(mx.add(a, b))

mx.metal.stop_capture()
