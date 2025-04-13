if st.button("🔄", key=f"regenerate_{index}", help="Regenerate response", use_container_width=False, format_func=lambda x: 'Regenerate Response'):
    user_query = st.session_state.chat_history[index - 1]["content"]

    with st.chat_message("assistant", avatar="🤖"):
        message_placeholder = st.empty()
        generating_response_text = "Generating response..."
        with st.spinner(generating_response_text):
            dynamic_placeholders = extract_dynamic_placeholders(user_query, nlp)
            response_gpt = generate_response(model, tokenizer, user_query)
            full_response = replace_placeholders(response_gpt, dynamic_placeholders, static_placeholders)
            # time.sleep(1) # Optional delay

        message_placeholder.markdown(full_response, unsafe_allow_html=True)
        st.session_state.chat_history[index]["content"] = full_response # Update chat history with new response
    st.rerun() # Rerun to display updated chat history
