import unittest
import torch
from torch import nn


class Boundary(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale=nn.Parameter(torch.tensor(10.))
    def forward(self,x):
        value=self.scale*(x.flatten(1).mean(1)-.5)
        return torch.stack([value,torch.zeros_like(value)],1)


class CWTests(unittest.TestCase):
    def test_margin_confidence_and_score_shift(self):
        from dataset3.cw import cw_margin
        z=torch.tensor([[4.,1.,2.],[0.,3.,1.]],dtype=torch.float64,requires_grad=True)
        y=torch.tensor([0,0])
        expected=torch.tensor([2.5,0.],dtype=torch.float64)
        torch.testing.assert_close(cw_margin(z,y,.5),expected)
        torch.testing.assert_close(cw_margin(z+123.,y,.5),expected)
        gradient,=torch.autograd.grad(cw_margin(z,y,.5).sum(),z)
        torch.testing.assert_close(gradient,torch.tensor([[1.,0.,-1.],[0.,0.,0.]],dtype=torch.float64))

    def test_known_minimum_l2_boundary_and_model_immutability(self):
        from dataset3.cw import cw_l2
        model=Boundary().eval()
        before={k:v.clone() for k,v in model.state_dict().items()}
        x=torch.tensor([[[[.7,.7]]]])
        adv,trace=cw_l2(model,x,torch.tensor([0]),steps=240,binary_search_steps=5,learning_rate=.05,initial_const=.01,abort_early=False)
        self.assertEqual(model(adv).argmax(1).item(),1)
        distance=(adv-x).square().sum().sqrt().item()
        self.assertGreaterEqual(distance,.2*2**.5-1e-5)
        self.assertLess(distance,.31)
        self.assertTrue(trace['found'][0])
        self.assertFalse(model.training)
        self.assertIsNone(model.scale.grad)
        for k,v in model.state_dict().items():self.assertTrue(torch.equal(v,before[k]))

    def test_project_box_blocks_unreachable_boundary_and_zero_budget(self):
        from dataset3.cw import cw_l2
        model=Boundary().eval()
        x=torch.tensor([[[[.7,.7]]]])
        for epsilon in [0.,.1]:
            adv,trace=cw_l2(model,x,torch.tensor([0]),epsilon=epsilon,steps=100,binary_search_steps=4,learning_rate=.05,abort_early=False)
            torch.testing.assert_close(adv,x,rtol=0,atol=0)
            self.assertFalse(trace['found'][0])
        adv,trace=cw_l2(model,x,torch.tensor([0]),epsilon=.3,steps=240,binary_search_steps=5,learning_rate=.05,initial_const=.01,abort_early=False)
        self.assertEqual(model(adv).argmax(1).item(),1)
        self.assertLessEqual((adv-x).abs().max().item(),.3+1e-6)

    def test_per_image_search_and_already_wrong_inputs(self):
        from dataset3.cw import cw_l2
        model=Boundary().eval()
        x=torch.tensor([[[[.55,.55]]],[[[.9,.9]]],[[[.3,.3]]]])
        labels=torch.zeros(3,dtype=torch.long)
        kw=dict(epsilon=.2,steps=160,binary_search_steps=5,learning_rate=.05,initial_const=.01,abort_early=False)
        a,trace=cw_l2(model,x,labels,**kw)
        b,_=cw_l2(model,x,labels,**kw)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertEqual(trace['found'],[True,False,True])
        self.assertTrue(torch.equal(a[2],x[2]))
        self.assertTrue(torch.equal(a[1],x[1]))
        self.assertTrue(torch.isfinite(a).all())
        self.assertTrue(((a>=0)&(a<=1)).all())


if __name__=='__main__':unittest.main()
