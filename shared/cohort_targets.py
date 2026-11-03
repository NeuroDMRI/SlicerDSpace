"""Reuse the subject CLI target semantics when planning an optional batch."""


def selected_stages(arguments):
    from shared.run_subject_pipeline_cli import build_parser, _resolve_targets, _resolve_microstructure_stages
    from shared.algorithm_selection import algorithms, BUNDLES
    # Parsing only: no environment checks, data conversion, weights or execution.
    args, _ = build_parser().parse_known_args(['--subject-root', '/__batch_planning__', *arguments])
    models = _resolve_microstructure_stages(args)
    args.run_free_water, args.run_noddi, args.run_dki = (model in models for model in ('freewater', 'noddi', 'dki'))
    if args.bundle_tractography_backend:
        args.tractography_backend = args.bundle_tractography_backend
    targets = set(_resolve_targets(args))
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
            stages.update('dti_tracking' if name == 'dti_det' else name for name in algorithms(args.tractography_backend))
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
    if 'csd_act' in stages:
        stages.update(('segmentation', 'parcellation', 'registered_dti'))
    if 'saf' in stages:
        stages.add('surface_postprocess')
    return stages
