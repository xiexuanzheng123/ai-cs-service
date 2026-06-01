from dataclasses import dataclass


@dataclass(frozen=True)
class RuleResult:
    intent: str
    risk_level: str
    transfer_to_human: bool
    reply: str
    suggestions: list[str]


def classify_message(message: str) -> RuleResult:
    normalized = message.strip().lower()

    if any(keyword in normalized for keyword in ["退款", "退钱", "退费", "退会员"]):
        return RuleResult(
            intent="refund",
            risk_level="high",
            transfer_to_human=True,
            reply="退款问题需要人工客服核实订单和支付状态，我已经为您转人工处理。",
            suggestions=["查看订单", "补充支付截图", "继续描述问题"],
        )

    if any(keyword in normalized for keyword in ["投诉", "举报客服", "我要举报", "不满意"]):
        return RuleResult(
            intent="complaint",
            risk_level="high",
            transfer_to_human=True,
            reply="投诉问题需要人工客服跟进处理，我已经为您转人工。",
            suggestions=["补充投诉原因", "上传截图", "继续描述问题"],
        )

    if any(keyword in normalized for keyword in ["注销", "删除账号", "改手机号", "换绑", "修改实名"]):
        return RuleResult(
            intent="account_change",
            risk_level="high",
            transfer_to_human=True,
            reply="账号资料变更或删除需要人工核验身份，我已经为您转人工处理。",
            suggestions=["准备账号信息", "准备身份校验资料", "继续描述问题"],
        )

    if any(keyword in normalized for keyword in ["密码", "登录", "登不上", "设备", "异常"]):
        return RuleResult(
            intent="account_problem",
            risk_level="medium",
            transfer_to_human=False,
            reply="您可以先尝试找回账号密码，或检查是否为异常设备登录。如果仍无法登录，我可以继续为您转人工处理。",
            suggestions=["找回账号密码", "账号异常设备登录", "转人工"],
        )

    if any(keyword in normalized for keyword in ["充值", "充了多少", "金额", "付款"]):
        return RuleResult(
            intent="recharge_query",
            risk_level="low",
            transfer_to_human=False,
            reply="充值金额需要通过业务系统查询。当前本地演示环境会由 Go 网关返回模拟充值记录。",
            suggestions=["查看充值记录", "会员问题", "转人工"],
        )

    if any(keyword in normalized for keyword in ["你好", "您好", "hello", "hi"]):
        return RuleResult(
            intent="greeting",
            risk_level="low",
            transfer_to_human=False,
            reply="您好，很高兴为您服务，请问有什么可以帮您？",
            suggestions=["账号问题", "会员问题", "功能咨询", "转人工"],
        )

    return RuleResult(
        intent="unknown",
        risk_level="medium",
        transfer_to_human=False,
        reply="这个问题我暂时没有完全理解。您可以换一种说法，或选择转人工继续处理。",
        suggestions=["账号问题", "会员问题", "功能咨询", "转人工"],
    )

