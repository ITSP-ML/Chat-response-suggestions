<script>
  import AutoComplete from "simple-svelte-autocomplete";
  let selectedCountry;

  async function get_suggestions_list(keyword) {
    // let x = JSON.stringify({ text: "P" });
    let data = { text: keyword };
    const url = "http://127.0.0.1:8008/sugg";
    const response = await fetch(url, {
      method: "POST",
      headers: {
        Accept: "application/json",
        "Content-Type": "application/json",
      },
      body: JSON.stringify(data),
    });
    return response.json();
  }
	import { onMount } from "svelte";
	import ChatMessage from "./ChatMessage.svelte";
	import TodayDivider from "./TodayDivider.svelte";
	import Fa from "svelte-fa";
	// import Api from "./Api.svelte";
	import {
		faUsers,
		faCompressArrowsAlt,
		faComments,
		faEnvelope,
	} from "@fortawesome/free-solid-svg-icons";
	let chat_id = 1;
	let nameMe = "Customer";
	let profilePicMe =
		"https://p0.pikist.com/photos/474/706/boy-portrait-outdoors-facial-men-s-young-t-shirt-hair-person-thumbnail.jpg";

	let nameChatPartner = "Agent";
	let profilePicChatPartner =
		"https://storage.needpix.com/rsynced_images/male-teacher-cartoon.jpg";
	let example_id = 17097374;
	let last_msg_id = 99999;
	let knowledge = "";
	let promise;
	let messages = [];

	$: if (example_id != -1) {
		promise = get_chat(example_id, last_msg_id, knowledge);
	}

	async function get_chat(example_id, last_msg_id, knowledge) {
		let thisexample_id = example_id;
		let data = { id: example_id, last_msg: last_msg_id}//, curr_knowledge: knowledge};
		const url = "http://127.0.0.1:8008/get_data";
		const res = await fetch(url, {
			method: "POST",
			headers: {
				Accept: "application/json",
				"Content-Type": "application/json",
			},
			body: JSON.stringify(data),
		});
		console.log(res);
		let predictions = await res.json();
		console.log(predictions);
		if (res.ok && thisexample_id == example_id) {
			messages = predictions.messages;

		}
	}
</script>

<svelte:head>
	<link
		rel="stylesheet"
		href="https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css"
	/>
</svelte:head>
<input type="int" placeholder=" example id" bind:value={example_id} />
<input
	type="int"
	placeholder=" stop in message number"
	bind:value={last_msg_id}
/>
<input
	type="text"
	placeholder="enter relevant knowledge"
	bind:value={knowledge}
/>
{#await promise}
	<div class="loading"><img messages="/loading.gif" alt="Loading..." /></div>
{/await}
<div class="card card-danger direct-chat direct-chat-danger">
	<div class="card-header">
		<div class="card-tools d-flex">
			<img
				class="contacts-img"
				src={profilePicChatPartner}
				alt="profilePic"
			/>
			<span class="contacts-name">{nameChatPartner}</span>
			<span class="mr-auto" />
			<button type="button" class="btn btn-tool" title="Contacts"
				><Fa icon={faUsers} /></button
			>
			<button type="button" class="btn btn-tool"
				><Fa icon={faCompressArrowsAlt} /></button
			>
		</div>
	</div>
	<div class="card-body">
		<div class="direct-chat-messages">
			{#each messages as message}
				<ChatMessage
					{nameMe}
					{profilePicMe}
					{nameChatPartner}
					{profilePicChatPartner}
					message={message.message}
					timestamp={message.messageIndex}
					sentByMe={message.sentByMe}
					timeRead={message.timeRead}
				/>
			{/each}
		</div>
	</div>
	<div class="card-footer">
		<div class="input-group">
			<!-- <input
				type="text"
				placeholder={similar_topics}
				class="form-control"
			/> -->
			<AutoComplete
			searchFunction={get_suggestions_list}
			bind:selectedItem={selectedCountry}
			labelFieldName="sugg"
			maxItemsToShowInList={10}
			delay={200}
			localFiltering={false}
			lowercaseKeywords={true}
			className  ="form-control"
			noInputStyles = {false}
			/>
			<span class="input-group-append">
				<button type="button" class="btn btn-primary">Send</button>
			</span>
		</div>
	</div>
</div>

<style>
	.direct-chat .card-body {
		overflow-x: hidden;
		padding: 0;
		position: relative;
	}

	.direct-chat-messages {
		-webkit-transform: translate(0, 0);
		transform: translate(0, 0);
		height: 800px;
		overflow: auto;
		padding: 10px;
		transition: -webkit-transform 0.5s ease-in-out;
		transition: transform 0.5s ease-in-out;
		transition: transform 0.5s ease-in-out,
			-webkit-transform 0.5s ease-in-out;
	}

	.contacts-img {
		border-radius: 50%;
		width: 40px;
		height: 40px;
	}
	.contacts-name {
		margin-left: 15px;
		font-weight: 600;
	}
</style>
