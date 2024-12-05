import xuance

if __name__ == '__main__':
    runner = xuance.get_runner(method='maddpg',
                            env='mpe',
                            env_id='simple_spread_v3',
                            is_test=False)
    runner.run()