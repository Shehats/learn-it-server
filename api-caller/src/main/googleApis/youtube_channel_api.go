package googleApis

import (
	"fmt"
	"log"
	"net/http"

	"google.golang.org/api/googleapi/transport"
	"google.golang.org/api/youtube/v3"
)

// YoutubeSearch stores return from youtube search
type YoutubeSearch struct {
	Videos    map[string]string
	Channels  map[string]string
	Playlists map[string]string
}

// CreateYoutubeClient returns youtube client using developerKey
func CreateYoutubeClient(developerKey string) *youtube.Service {
	client := &http.Client{
		Transport: &transport.APIKey{Key: developerKey},
	}
	service, err := youtube.New(client)

	if err != nil {
		log.Fatalf("Error creating new YouTube client: %v", err)
	}
	return service
}

// SearchYoutube querys youtube for strings
func SearchYoutube(service *youtube.Service, searchStr string) YoutubeSearch {
	resp, err := service.Search.List("id,snippet").Q(searchStr).Do()
	if err != nil {
		log.Panicf(fmt.Sprintf("Error fetching %v with error: %v", searchStr, err))
		// Return empty response
		return YoutubeSearch{}
	}
	videos := make(map[string]string)
	channels := make(map[string]string)
	playlists := make(map[string]string)

	// Iterate through each item and add it to the correct list.
	for _, item := range resp.Items {
		switch item.Id.Kind {
		case "youtube#video":
			videos[item.Id.VideoId] = item.Snippet.Title
		case "youtube#channel":
			channels[item.Id.ChannelId] = item.Snippet.Title
		case "youtube#playlist":
			playlists[item.Id.PlaylistId] = item.Snippet.Title
		}
	}
	return YoutubeSearch{Videos: videos, Channels: channels, Playlists: playlists}
}
