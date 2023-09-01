import torch
from torch import nn
import time

class TBPTT():
    def __init__(self, one_step_module, loss_module, k1, k2, optimizer):
        self.one_step_module = one_step_module
        self.loss_module = loss_module
        self.k1 = k1
        self.k2 = k2
        self.retain_graph = k1 < k2
        # You can also remove all the optimizer code here, and the
        # train function will just accumulate all the gradients in
        # one_step_module parameters
        self.optimizer = optimizer

    def train(self, input_sequence, init_state):
        def state_detach(state):
            state_after_detach = {}
            for k, v in state.items():
                state_after_detach[k] = v.detach()
                state_after_detach[k].requires_grad = True
            return state_after_detach
        def get_states_grad(states):
            states_grad = {}
            for k, v in states.items():
                states_grad[k] = v.grad
            return states_grad
        def states_backward(states, current_grad):
            for k, v in states.items():
                v.backward(current_grad[k], retain_graph=True)
        states = [(None, init_state)]
        outputs = []
        targets = []
        for j in range(input_sequence[0].shape[1]):
            inp = input_sequence[0][:,j,:]
            target = input_sequence[1][:,j,:]
            # state = states[-1][1].detach()  # 每个state都是独立的
            # state.requires_grad = True
            state = state_detach(states[-1][1])
            output, new_state = self.one_step_module(inp, state)
            outputs.append(output)
            targets.append(target)
            states.append((state, new_state))
            while len(outputs) > self.k1:
                del outputs[0]
                del targets[0]

            while len(states) > self.k2:
                # Delete stuff that is too old
                del states[0]

            if (j + 1) % self.k1 == 0:  # 直到前向传播足够步数后
                # loss = self.loss_module(output, target) # loss对应梯度存储在states[-1][0],由于输入detach过，虽然states[-2][1]的值与它一样，但没有grad
                self.optimizer.zero_grad()
                # backprop last module (keep graph only if they ever overlap)
                # start = time.time()
                # loss.backward(retain_graph=self.retain_graph)
                for i in range(self.k2 - 1):  # 求loss本身就求了一个时间步长的梯度，所以只需要往前求k2-1个步长
                    if i < self.k1:
                        # loss = self.loss_module(outputs[-i - 1], targets[-i - 1])
                        loss = self.one_step_module.get_loss()
                        print(f"loss:{loss.item()}")
                        loss.backward(retain_graph=True)
                        # g = make_dot(loss)
                        # g.render(filename='netStructure/myNetModel', view=False, format='pdf')
                    # if we get all the way back to the "init_state", stop
                    if states[-i - 2][0] is None:
                        break
                    current_grad = get_states_grad(states[-i - 1][0])  # 求末尾梯度
                    states_backward(states[-i - 2][1], current_grad)   # 传播到上一个state中，grad将会存储到states[-i-2][0]中
                # print("backward time: {}".format(time.time() - start))
        print("梯度裁剪后进行step")
        torch.nn.utils.clip_grad_norm_(self.one_step_module.parameters(), 0.9)
        self.optimizer.step()


class MyMod(nn.Module):
    def __init__(self):
        super(MyMod, self).__init__()
        self.lin = nn.Linear(2 * layer_size, 2 * layer_size)

    def forward(self, inp, state):
        global idx
        full_out = self.lin(torch.cat([inp, state], 1))
        # out, new_state = full_out.chunk(2, dim=1)
        out = full_out.narrow(1, 0, layer_size)
        new_state = full_out.narrow(1, layer_size, layer_size)

        def get_pr(idx_val):
            def pr(*args):
                print("doing backward {}".format(idx_val))
            return pr

        new_state.register_hook(get_pr(idx))
        out.register_hook(get_pr(idx))
        print("doing fw {}".format(idx))
        idx += 1
        return out, new_state


if __name__ == "__main__":
    seq_len = 20
    layer_size = 50

    idx = 0
    one_step_module = MyMod()
    loss_module = nn.MSELoss()
    input_sequence = [(torch.rand(200, layer_size), torch.rand(200, layer_size))] * seq_len

    optimizer = torch.optim.SGD(one_step_module.parameters(), lr=1e-3)

    runner = TBPTT(one_step_module, loss_module, 10, 10, optimizer)

    runner.train(input_sequence, torch.zeros(200, layer_size))
    print("done")
