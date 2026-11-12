"""Training on several GPUs of one machine: one process a GPU (`torchrun`),
each on its own clips, their gradients averaged before every step, so every
copy of the weights takes the same steps and stays the same."""

import datetime
import os

import torch
import torch.distributed as dist


class Gpus:
    """`rank` of `world` processes; one alone without `torchrun`. Only rank 0
    (`lead`) validates, saves and logs."""

    def __init__(self) -> None:
        self.world = int(os.environ.get("WORLD_SIZE", 1))
        self.rank = int(os.environ.get("RANK", 0))
        self.lead = self.rank == 0
        if self.world > 1:
            torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
            # Rank 0 validates for minutes while the others wait.
            dist.init_process_group("nccl", timeout=datetime.timedelta(hours=1))

    def same_start(self, model) -> None:
        """Every copy starts from rank 0's weights (the new ones are drawn at random)."""
        if self.world > 1:
            for t in model.state_dict().values():
                dist.broadcast(t, 0)

    def average_grads(self, model) -> None:
        """Each weight's gradient averaged over the processes, in one exchange;
        a weight one process left without a gradient counts as zero there."""
        if self.world == 1:
            return
        ps = [p for p in model.parameters() if p.requires_grad]
        flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten().float() for p in ps])
        dist.all_reduce(flat)
        flat /= self.world
        at = 0
        for p in ps:
            n = p.numel()
            p.grad = flat[at:at + n].view_as(p).to(p.dtype)
            at += n

    def check_same(self, model) -> None:
        """Stop if the copies of the weights have come apart."""
        if self.world == 1:
            return
        s = torch.stack([p.detach().float().sum() for p in model.parameters()]).sum()
        hi, lo = s.clone(), s.clone()
        dist.all_reduce(hi, op=dist.ReduceOp.MAX)
        dist.all_reduce(lo, op=dist.ReduceOp.MIN)
        if hi != lo:
            raise RuntimeError(f"the GPUs' weights differ: checksums {lo.item()} to {hi.item()}")

    def gather(self, obj):
        """Every process's `obj`, in rank order, on every process."""
        if self.world == 1:
            return [obj]
        out = [None] * self.world
        dist.all_gather_object(out, obj)
        return out

    def wait(self) -> None:
        """Every process here before any goes on (around rank 0's validation)."""
        if self.world > 1:
            dist.barrier()

    def close(self) -> None:
        if self.world > 1:
            dist.destroy_process_group()
