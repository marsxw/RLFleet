
def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def test_agent(env_test, agent, logger=None,   n=5,  render=False):
    for j in range(n):
        o, r, d, ep_ret, ep_len = env_test.reset(), 0, False, 0, 0
        while d == False or ep_len < env_test._max_episode_steps:
            a = agent.select_action(o,  evaluate=True)
            o, r, d, _ = env_test.step(a)
            ep_ret += r
            ep_len += 1

            if render and ep_len % 100 == 0:
                env_test.render()

        if logger is not None:
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
