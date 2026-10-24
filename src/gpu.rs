//! CUDA device training. Errors never fall back to CPU execution.
use crate::{
    invalid,
    training::{GpuEvidence, Observation, TrainingResult},
    Result,
};
use cudarc::{
    driver::{
        sys::{CUdevice_attribute, CUevent_flags},
        CudaContext, LaunchConfig, PushKernelArg,
    },
    nvrtc::{compile_ptx_with_opts, CompileOptions},
};

// Each of the 64 CUDA threads owns one training and one validation example.
// Device barriers keep all gradient, parameter and loss computations on GPU.
const KERNEL: &str = r#"
extern "C" __global__ void fit_linear(const double* x, const double* y,
 int ntrain, int nval, double rate, int epochs, double* coefficients, double* history) {
  __shared__ double w, b, gx[64], gb[64], losses[64];
  const int i = threadIdx.x;
  if (i == 0) { w=0.0; b=0.0; }
  __syncthreads();
  for (int epoch=0; epoch<epochs; ++epoch) {
    double sw_local=0.0, sb_local=0.0;
    for(int k=i;k<ntrain;k+=64) { double error=w*x[k]+b-y[k]; sw_local+=error*x[k]; sb_local+=error; }
    gx[i]=sw_local; gb[i]=sb_local;
    __syncthreads();
    if (i==0) {
      double sw=0.0, sb=0.0;
      for(int j=0;j<64;++j) { sw+=gx[j]; sb+=gb[j]; }
      w-=rate*2.0*sw/ntrain;
      b-=rate*2.0*sb/ntrain;
    }
    __syncthreads();
    double local_loss=0.0;
    for(int k=i;k<nval;k+=64) { double residual=w*x[ntrain+k]+b-y[ntrain+k]; local_loss+=residual*residual; }
    losses[i]=local_loss;
    __syncthreads();
    if (i==0) {
      double loss=0.0;
      for(int j=0;j<64;++j) loss+=losses[j];
      history[epoch]=loss/nval;
    }
    __syncthreads();
  }
  if(i==0) { coefficients[0]=w; coefficients[1]=b; }
}
"#;
fn cuda_error(error: impl std::fmt::Debug) -> crate::Error {
    invalid(format!("CUDA execution failed: {error:?}"))
}

/// Execute actual gradient updates and validation calculations on CUDA device 0.
/// Returned values must match the independently computed CPU reference.
pub fn train_cuda(rate: f64, epochs: usize) -> Result<TrainingResult> {
    train_cuda_on(rate, epochs, &crate::dataset::embedded()?)
}
/// Execute real CUDA training on supplied samples, checked against the CPU reference.
pub fn train_cuda_on(rate: f64, epochs: usize, examples: &[[f64; 2]]) -> Result<TrainingResult> {
    crate::dataset::validate(examples)?;
    if !rate.is_finite() || rate <= 0. || rate > 0.2 || epochs == 0 || epochs > 10000 {
        return Err(invalid(
            "CUDA training needs rate in (0, 0.2] and 1..10000 epochs",
        ));
    }
    let context = CudaContext::new(0).map_err(cuda_error)?;
    let name = context.name().map_err(cuda_error)?;
    let major = context
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
        .map_err(cuda_error)?;
    let minor = context
        .attribute(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
        .map_err(cuda_error)?;
    let stream = context.default_stream();
    let ptx = compile_ptx_with_opts(
        KERNEL,
        CompileOptions {
            fmad: Some(false),
            arch: Some("compute_80"),
            ..Default::default()
        },
    )
    .map_err(cuda_error)?;
    let module = context.load_module(ptx).map_err(cuda_error)?;
    let function = module.load_function("fit_linear").map_err(cuda_error)?;
    let x: Vec<f64> = examples.iter().map(|p| p[0]).collect();
    let y: Vec<f64> = examples.iter().map(|p| p[1]).collect();
    let x = stream.memcpy_stod(&x).map_err(cuda_error)?;
    let y = stream.memcpy_stod(&y).map_err(cuda_error)?;
    let mut coefficients = stream.alloc_zeros::<f64>(2).map_err(cuda_error)?;
    let mut history = stream.alloc_zeros::<f64>(epochs).map_err(cuda_error)?;
    let count = epochs as i32;
    let ntrain = (examples.len() / 2) as i32;
    let nval = examples.len() as i32 - ntrain;
    let start = stream
        .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(cuda_error)?;
    let mut launch = stream.launch_builder(&function);
    launch
        .arg(&x)
        .arg(&y)
        .arg(&ntrain)
        .arg(&nval)
        .arg(&rate)
        .arg(&count)
        .arg(&mut coefficients)
        .arg(&mut history);
    // SAFETY: fit_linear receives examples.len() f64 inputs in each array, two
    // coefficient outputs and `epochs` history outputs. ntrain+nval equals
    // the input length; strided loops check their bounds. One block of 64 threads
    // matches the kernel's shared arrays and barriers. All allocations remain
    // alive and are bound to this same CUDA stream until synchronization/copy.
    unsafe {
        launch.launch(LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map_err(cuda_error)?;
    let end = stream
        .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        .map_err(cuda_error)?;
    end.synchronize().map_err(cuda_error)?;
    let elapsed = start.elapsed_ms(&end).map_err(cuda_error)? as f64;
    let coefficients = stream.memcpy_dtov(&coefficients).map_err(cuda_error)?;
    let losses = stream.memcpy_dtov(&history).map_err(cuda_error)?;
    let reference = crate::training::train_on(rate, epochs, examples)?;
    let mut error = (coefficients[0] - reference.weight)
        .abs()
        .max((coefficients[1] - reference.bias).abs());
    for (loss, expected) in losses.iter().zip(&reference.history) {
        if !loss.is_finite() {
            return Err(invalid("CUDA returned nonfinite training loss"));
        }
        error = error.max((loss - expected.val_loss).abs());
    }
    if !coefficients.iter().all(|v| v.is_finite())
        || !elapsed.is_finite()
        || elapsed <= 0.
        || !error.is_finite()
        || error > 1e-10
    {
        return Err(invalid(
            "GPU training failed timing/finite-value/CPU-parity verification",
        ));
    }
    let final_loss = *losses.last().ok_or_else(|| invalid("Missing CUDA loss"))?;
    Ok(TrainingResult {
        learning_rate: rate,
        epochs,
        weight: coefficients[0],
        bias: coefficients[1],
        val_loss: final_loss,
        history: losses
            .into_iter()
            .enumerate()
            .map(|(i, val_loss)| Observation {
                epoch: i + 1,
                val_loss,
            })
            .collect(),
        execution: "GPU, Rust CUDA; device training verified".into(),
        gpu: Some(GpuEvidence {
            device_name: name,
            compute_capability: [major, minor],
            kernel_launches: 1,
            cuda_elapsed_ms: elapsed,
            cpu_reference_max_abs_error: error,
        }),
    })
}
