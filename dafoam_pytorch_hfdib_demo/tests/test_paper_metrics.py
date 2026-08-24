import numpy as np
from tpfm_reference.paper_metrics import (paper_figure6_normalize, paper_style_metrics,
                                           physical_metrics, total_variation)


def test_paper_normalization_uses_prediction_limits():
    prediction=np.array([[2.,4.],[3.,2.]])
    reference=np.array([[1.,5.],[3.,2.]])
    p,r,lower,upper=paper_figure6_normalize(prediction,reference)
    assert lower==2 and upper==4
    np.testing.assert_allclose(p,[[0,1],[.5,0]])
    np.testing.assert_allclose(r,[[-.5,1.5],[.5,0]])


def test_reconstructed_tv_and_mse_tv():
    field=np.array([[0.,1.],[0.,1.]])
    assert total_variation(field)==1.0
    metrics=paper_style_metrics(field,np.zeros_like(field))
    assert metrics["mse"]==.5
    assert metrics["mse_tv"]==.6


def test_physical_metrics_gauge_pressure():
    u=np.ones((2,2,2)); p=np.array([[3.,2.],[3.,2.]])
    result=physical_metrics(u,u,p+10,p,inlet_mask=np.array([[1,0],[1,0]],dtype=bool),outlet_mask=np.array([[0,1],[0,1]],dtype=bool))
    assert result["relative_velocity_error"]==0
    assert result["relative_pressure_error"]<1e-14
    assert abs(result["pressure_drop_ratio"]-1)<1e-14
