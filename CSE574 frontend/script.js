var img1_url = "";
var img2_url = "";
var fake = "";
var model_fake = "";
var model_prob = "";
var chosen = false;

$('.collapsible').collapsible({
	accordion: true,    
	onOpenStart: function(e){
		$(e).find('.collapsible-header div:last-child i').text('keyboard_arrow_up');

	},
	onCloseStart:function(e){
		$(e).find('.collapsible-header div:last-child i').text('keyboard_arrow_down');
	}
});

$.ajax({
	url: 'http://localhost:5000/get_predicitons', // replace with link to API that produces a JSON
	async: false,
	dataType: 'json',
	success: function(response){
		img1_url = response.img1;
		img2_url = response.img2;
		fake = response.fake;
		model_fake = response.model.fake;
		model_prob = response.model.probability;
	}
});

$("#local-input-img img").attr('src',img1_url+'?t=' + new Date().getTime());
$("#local-depth-img img").attr('src',img2_url+'?t=' + new Date().getTime());

function getPath(img){
	if(img==1)
		return img1_url;
	return img2_url;
}


function select(img){

	if(chosen) return;
	chosen = true;

	if(fake == "img1"){
		$("#person1").html('<span style="color: #0f9d58;">Fake</span>');
		$("#person2").html('<span>Real</span>');
	}
	else{
		$("#person2").html('<span style="color: #0f9d58;">Fake</span>');
		$("#person1").html('<span>Real</span>');
	}

	if(fake == "img1" && img == 1 || fake == "img2" && img == 2){
		$("#answer").html('<i class="material-icons left" style="color: #0f9d58;">check_circle</i>');
		$("#answered").html('Correct!');
		$("#answer_body").html('The fake person is on the ' + (fake == "img1" ? 'left.' : 'right.'));
		$('.collapsible').collapsible('open', 0);
	}
	else{
		$("#answer").html('<i class="material-icons left">close</i>');
		$("#answered").html('Wrong!');
		$("#answer_body").html('The fake person is on the ' + (fake == "img1" ? 'left.' : 'right.'));
		$('.collapsible').collapsible('open', 0);
	}

	setTimeout(function(){
		$("#model-body").html('The Model predicted that the person on the <span style="color: #0f9d58;">' + (model_fake == "img1" ? 'left' : 'right') + '</span> is fake, with a probability: <span style="color: #0f9d58;">' + model_prob + "</span>");
		$('.collapsible').collapsible('open', 1);
	}, 2500);
}