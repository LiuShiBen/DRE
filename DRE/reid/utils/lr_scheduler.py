from .cosine_lr import CosineLRScheduler

def create_scheduler(optimizer, epochs, lr):

    lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=epochs,
            lr_min=0.002 * lr,
            t_mul= 1.,
            decay_rate=0.1,
            warmup_lr_init=0.01 * lr,
            warmup_t=5,
            cycle_limit=1,
            t_in_epochs=True,
            noise_range_t=None,
            noise_pct= 0.67,
            noise_std= 1.,
            noise_seed=42,
        )

    return lr_scheduler