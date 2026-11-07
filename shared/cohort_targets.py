"""Reuse the subject CLI target semantics when planning an optional batch."""


def selected_stages(arguments):
    from shared.run_subject_pipeline_cli import build_parser, resolve_analysis_options
    from shared.algorithm_selection import algorithms, BUNDLES
    from shared.user_config import explicit_cli_destinations, coupled_explicit_cli_destinations
    # Parsing only: no environment checks, data conversion, weights or execution.
    parser = build_parser()
    args, _ = parser.parse_known_args(['--subject-root', '/__batch_planning__', *arguments])
    explicit = coupled_explicit_cli_destinations(explicit_cli_destinations(parser, arguments))
    targets = set(resolve_analysis_options(args, explicit_dests=explicit))
    stages = set()
    for target in targets:
        if target == 'validation':
            continue
        if target == 'microstructure':
            stages.add('microstructure_dti')
        elif target.startswith('microstructure_'):
            stages.add('microstructure_mkcurve' if target == 'microstructure_dki'
                       and args.dki_fit_method.startswith('MKCURVE') else target)
            stages.add('microstructure_dti')
        elif target == 'tractography':
            tracking = getattr(args, 'closure_tractography_backend', None) or args.tractography_backend
            stages.update('dti_tracking' if name == 'dti_det' else name for name in algorithms(tracking))
        elif target == 'bundles':
            stages.update(algorithms(args.bundle_backend, BUNDLES, 'wmql'))
        else:
            stages.add(target)
    # These are possible preparation nodes, not a requirement to recompute them.
    if stages & {'preprocessing', 'segmentation', 'parcellation', 'surface', 'csd_act', 'ukf', 'dti_tracking', 'saf'} or any(
            stage.startswith('microstructure_') for stage in stages):
        stages.update(('brain_mask', 'native_dti'))
    if stages & {'parcellation', 'surface'}:
        stages.add('registered_dti')
    # Prerequisite targets (segmentation, parcellation, surface_postprocess, ...) are already in
    # `targets`: resolve_analysis_options returns the closure of shared/target_closure.py.
    return stages
