"""
Streamlit监控面板
运行：streamlit run dashboard/monitor.py
"""
import streamlit as st
import redis
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config(page_title="RAG平台监控", layout="wide")

# 连接Redis
r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

st.title(" RAG平台监控面板")

# 顶部指标卡片
col1, col2, col3, col4 = st.columns(4)

with col1:
    # 统计活跃会话数
    active_sessions = len(r.keys("chat:history:*"))
    st.metric("活跃会话数", active_sessions)

    with col2:
        cache_count = len(r.keys("cache:rag:*"))
        st.metric("缓存条目数", cache_count)

        with col3:
            st.metric("Redis状态", " 正常" if r.ping() else " 异常")

            with col4:
                st.metric("当前时间", datetime.now().strftime("%H:%M:%S"))

                st.divider()

                # 缓存命中率（模拟数据，实际从日志统计）
                st.subheader("缓存命中率趋势")
                dates = [(datetime.now() - timedelta(days=i)).strftime("%m-%d") for i in range(7)][::-1]
                hit_rates = [65, 70, 68, 72, 75, 73, 78]

                df = pd.DataFrame({"日期": dates, "命中率(%)": hit_rates})
                fig = px.line(df, x="日期", y="命中率(%)", title="近7天缓存命中率")
                st.plotly_chart(fig, use_container_width=True)

                st.divider()

                # 最近日志
                st.subheader("最近请求日志")
                log_file = "./logs/app.log"
            try:
                with open(log_file, "r") as f:
                    lines = f.readlines()[-20:] # 最后20行
                    st.code("".join(lines))
            except FileNotFoundError:
                st.info("暂无日志数据，启动服务后会自动生成")
