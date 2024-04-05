import neurogym as ngym

def main():
    # All supervised tasks:
    tasks = (
        # 'ContextDecisionMaking-v0', 
        # 'DelayComparison-v0', 
        # 'DelayMatchCategory-v0',
        # 'DelayMatchSample-v0',
        # 'DelayMatchSampleDistractor1D-v0',
        # 'DelayPairedAssociation-v0',
        # 'DualDelayMatchSample-v0',
        # 'GoNogo-v0',
        # 'HierarchicalReasoning-v0',
        # 'IntervalDiscrimination-v0',
        # 'MotorTiming-v0',
        # 'MultiSensoryIntegration-v0',
        # 'OneTwoThreeGo-v0',
        # 'PerceptualDecisionMaking-v0',
        # 'PerceptualDecisionMakingDelayResponse-v0',
        'ProbabilisticReasoning-v0',
        # 'PulseDecisionMaking-v0',
        # 'ReachingDelayResponse-v0', # Different input type, so omitted
        # 'ReadySetGo-v0',
        # 'SingleContextDecisionMaking-v0',
    )

    kwargs = {'dt': 100}
    seq_len = 100

    datasets_params = []

    for task_idx, task in enumerate(tasks):

        # Make supervised dataset
        dataset = ngym.Dataset(task, env_kwargs=kwargs, batch_size=16,
                            seq_len=seq_len)
        
        dataset_params = {
            'dataset_name': task,
            'seq_length': seq_len,
            'dt': kwargs['dt'],
            'dataset': dataset,
            'convert_inputs': True, # Remap inputs through a random matrix + bias
            'input_dim': 10,
        }

        env = dataset.env
        ob_size = env.observation_space.shape[0]
        act_size = env.action_space.n

        print('Task {}: {}'.format(task_idx, task))
        print('  Observation size: {}, Action size: {}'.format(ob_size, act_size))

        datasets_params.append(dataset_params)


if __name__ == "__main__":
    main()